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
"""


@dataclass(frozen=True)
class SearchHit:
    sha: str
    snippet: str
    score: float
    occurrences: list[tuple[str, int, str]]  # (session, seq, kind)
    n_occurrences: int


class SearchIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("PRAGMA journal_mode=WAL;" + SCHEMA)

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
                r[0] for r in self.db.execute(
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
                        "INSERT INTO blob_fts(content, sha) VALUES (?, ?)", (content, sha)
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
            occ_sql = "SELECT session, seq, kind FROM items WHERE sha = ?"
            args: list = [sha]
            if kind:
                occ_sql += " AND kind = ?"
                args.append(kind)
            if session_like:
                occ_sql += " AND session LIKE ?"
                args.append(f"%{session_like}%")
            occ = self.db.execute(occ_sql + " ORDER BY session, seq", args).fetchall()
            if not occ:
                continue  # all occurrences excluded by filters
            hits.append(SearchHit(sha, snippet, score, occ[:3], len(occ)))
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
