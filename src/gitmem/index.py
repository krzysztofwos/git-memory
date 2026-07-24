"""Full-text search index over a gitmem store.

SQLite FTS5, keyed by blob SHA: content addressing means a blob is indexed
exactly once, ever — incremental ingestion never re-indexes anything. The
`items` table maps each indexed blob to every (session, seq) where it
appears, so a search hit carries provenance back to the exact context
position(s), and `gitmem show <sha>` returns the verbatim original.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs(
  sha TEXT PRIMARY KEY,
  nbytes INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS blob_fts USING fts5(content, sha UNINDEXED);
CREATE TABLE IF NOT EXISTS items(
  session TEXT NOT NULL,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  role TEXT NOT NULL,
  sha TEXT NOT NULL,
  PRIMARY KEY (session, seq)
);
CREATE INDEX IF NOT EXISTS items_sha ON items(sha);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings(
  sha TEXT NOT NULL,
  chunk_no INTEGER NOT NULL,
  offset INTEGER NOT NULL,
  model TEXT NOT NULL,
  vec BLOB NOT NULL,
  PRIMARY KEY (sha, chunk_no, model)
);
"""

RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass(frozen=True)
class SearchHit:
    sha: str
    snippet: str | None
    score: float
    occurrences: list[tuple[str, int, str]]  # (session, seq, kind)
    n_occurrences: int
    origin: str = "fts"  # fts | sem | both
    offset: int = 0  # chunk offset of best semantic match (snippet source)


class SearchIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("PRAGMA journal_mode=WAL;" + SCHEMA)
        self._matrix_cache: dict = {}

    def close(self) -> None:
        self.db.close()

    # ---- write ----

    def max_seq(self, session: str) -> int:
        row = self.db.execute(
            "SELECT MAX(seq) FROM items WHERE session = ?", (session,)
        ).fetchone()
        return row[0] or 0

    def missing_blobs(self, shas: set[str]) -> set[str]:
        known = set()
        shas = list(shas)
        for i in range(0, len(shas), 500):
            chunk = shas[i : i + 500]
            marks = ",".join("?" * len(chunk))
            known.update(
                r[0]
                for r in self.db.execute(
                    f"SELECT sha FROM blobs WHERE sha IN ({marks})", chunk
                )
            )
        return set(shas) - known

    def add(
        self,
        session: str,
        rows: list[tuple[int, str, str, str]],  # (seq, kind, role, sha)
        contents: dict[str, str],  # sha -> content, for blobs not yet indexed
    ) -> None:
        with self.db:
            for sha, content in contents.items():
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO blobs(sha, nbytes) VALUES (?, ?)",
                    (sha, len(content.encode())),
                )
                if cur.rowcount:
                    self.db.execute(
                        "INSERT INTO blob_fts(content, sha) VALUES (?, ?)",
                        (content, sha),
                    )
            self.db.executemany(
                "INSERT OR REPLACE INTO items(session, seq, kind, role, sha) "
                "VALUES (?, ?, ?, ?, ?)",
                [(session, seq, kind, role, sha) for seq, kind, role, sha in rows],
            )

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
            )

    # ---- vectors ----

    def missing_vectors(self, shas: set[str], model: str) -> set[str]:
        have = set()
        shas = list(shas)
        for i in range(0, len(shas), 500):
            chunk = shas[i : i + 500]
            marks = ",".join("?" * len(chunk))
            have.update(
                r[0]
                for r in self.db.execute(
                    f"SELECT DISTINCT sha FROM embeddings WHERE model = ? AND sha IN ({marks})",
                    [model, *chunk],
                )
            )
        return set(shas) - have

    def add_vectors(
        self, model: str, rows: list[tuple[str, int, int, "np.ndarray"]]
    ) -> None:
        """rows: (sha, chunk_no, offset, normalized float32 vector)."""
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings(sha, chunk_no, offset, model, vec) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (sha, no, off, model, vec.astype(np.float16).tobytes())
                    for sha, no, off, vec in rows
                ],
            )
        self._matrix_cache.pop(model, None)

    def vector_count(self, model: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
        ).fetchone()[0]

    def _matrix(self, model: str):
        if model not in self._matrix_cache:
            rows = self.db.execute(
                "SELECT sha, offset, vec FROM embeddings WHERE model = ?", (model,)
            ).fetchall()
            if not rows:
                self._matrix_cache[model] = ([], None)
            else:
                keys = [(sha, off) for sha, off, _ in rows]
                mat = (
                    np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float16)
                    .reshape(len(rows), -1)
                    .astype(np.float32)
                )
                self._matrix_cache[model] = (keys, mat)
        return self._matrix_cache[model]

    def semantic_ranked(
        self, qvec: "np.ndarray", model: str, limit: int
    ) -> list[tuple[str, int, float]]:
        """Brute-force cosine top-k, deduped to (sha, best offset, score)."""
        keys, mat = self._matrix(model)
        if mat is None:
            return []
        sims = mat @ qvec.astype(np.float32)
        order = np.argsort(sims)[::-1]
        out, seen = [], set()
        for i in order:
            sha, off = keys[i]
            if sha in seen:
                continue
            seen.add(sha)
            out.append((sha, off, float(sims[i])))
            if len(out) >= limit:
                break
        return out

    # ---- read ----

    def search(
        self,
        query: str,
        limit: int = 10,
        kind: str | None = None,
        session_like: str | None = None,
    ) -> list[SearchHit]:
        try:
            rows = self._match(query, limit * 4)
        except sqlite3.OperationalError:
            # Raw FTS5 syntax error (unbalanced quotes etc.) -> literal phrase.
            rows = self._match('"' + query.replace('"', '""') + '"', limit * 4)
        hits = []
        for sha, snippet, score in rows:
            occ = self._occurrences(sha, kind, session_like)
            if not occ:
                continue  # all occurrences excluded by filters
            hits.append(SearchHit(sha, snippet, score, occ[:3], len(occ)))
            if len(hits) >= limit:
                break
        return hits

    def _occurrences(self, sha, kind, session_like):
        occ_sql = "SELECT session, seq, kind FROM items WHERE sha = ?"
        args: list = [sha]
        if kind:
            occ_sql += " AND kind = ?"
            args.append(kind)
        if session_like:
            occ_sql += " AND session LIKE ?"
            args.append(f"%{session_like}%")
        return self.db.execute(occ_sql + " ORDER BY session, seq", args).fetchall()

    def hybrid_search(
        self,
        query: str,
        qvec: "np.ndarray | None",
        model: str,
        limit: int = 10,
        kind: str | None = None,
        session_like: str | None = None,
    ) -> list[SearchHit]:
        """Reciprocal-rank fusion of the FTS and semantic legs.

        Exact identifiers win via bm25; vague paraphrases win via vectors.
        Either leg may be empty (no vectors yet / no keyword overlap)."""
        pool = limit * 4
        try:
            fts = self._match(query, pool)
        except sqlite3.OperationalError:
            fts = self._match('"' + query.replace('"', '""') + '"', pool)
        sem = self.semantic_ranked(qvec, model, pool) if qvec is not None else []

        fused: dict[str, float] = {}
        snippets = {sha: snip for sha, snip, _ in fts}
        offsets = {sha: off for sha, off, _ in sem}
        for rank, (sha, _, _) in enumerate(fts):
            fused[sha] = fused.get(sha, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (sha, _, _) in enumerate(sem):
            fused[sha] = fused.get(sha, 0.0) + 1.0 / (RRF_K + rank)

        hits = []
        for sha, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            occ = self._occurrences(sha, kind, session_like)
            if not occ:
                continue
            origin = (
                "both"
                if sha in snippets and sha in offsets
                else "fts" if sha in snippets else "sem"
            )
            hits.append(
                SearchHit(
                    sha,
                    snippets.get(sha),
                    score,
                    occ[:3],
                    len(occ),
                    origin=origin,
                    offset=offsets.get(sha, 0),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _match(self, query: str, limit: int) -> list[tuple[str, str, float]]:
        return self.db.execute(
            "SELECT sha, snippet(blob_fts, 0, '>>>', '<<<', ' … ', 24), bm25(blob_fts) "
            "FROM blob_fts WHERE blob_fts MATCH ? ORDER BY bm25(blob_fts) LIMIT ?",
            (query, limit),
        ).fetchall()

    def counts(self) -> dict[str, int]:
        return {
            "blobs": self.db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0],
            "items": self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0],
            "sessions": self.db.execute(
                "SELECT COUNT(DISTINCT session) FROM items"
            ).fetchone()[0],
        }
