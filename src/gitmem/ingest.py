"""Incremental ingestion of Claude Code transcripts into a gitmem store.

Resumable at the item level: extraction is deterministic and append-only,
and a session's `next_seq - 1` is exactly the number of items already
ingested, so a live session is resumed by appending `events[ingested:]`.
Unchanged files are skipped via an mtime watermark kept in the index.

The git write phase can run across processes (branches don't contend).
The index update phase then derives rows from the store itself, so workers
never touch SQLite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from gitmem.core import ITEM_RE, MemoryStore
from gitmem.embed import chunks
from gitmem.index import SearchIndex
from gitmem.transcript import DEFAULT_ROOT, discover, extract_events, slug

MTIME_KEY = "last_ingest_mtime"


@dataclass
class IngestStats:
    files_seen: int = 0
    files_ingested: int = 0
    items_appended: int = 0
    items_indexed: int = 0
    blobs_indexed: int = 0
    blobs_embedded: int = 0
    seconds: float = 0.0
    sessions: list[str] = field(default_factory=list)


def ingest_file(store: MemoryStore, path: Path, root: Path) -> tuple[str, int]:
    """Append this transcript's new events. Returns (session, appended)."""
    events = extract_events(path)
    session = store.session(slug(path, root))
    ingested = session.next_seq - 1
    new = events[ingested:]
    for kind, role, content in new:
        session.append(kind, role, content)
    return session.name, len(new)


def _worker(args: tuple[str, str, str]) -> tuple[str, int]:
    store_path, path, root = args
    return ingest_file(MemoryStore(store_path), Path(path), Path(root))


def embed_blobs(index: SearchIndex, embedder, contents: dict[str, str]) -> int:
    """Chunk + embed blobs and store their vectors. Returns blobs embedded."""
    todo = index.missing_vectors(set(contents), embedder.name)
    if not todo:
        return 0
    rows, texts = [], []
    for sha in sorted(todo):
        for no, (off, piece) in enumerate(chunks(contents[sha])):
            rows.append((sha, no, off))
            texts.append(piece)
    vecs = embedder.embed_docs(texts)
    index.add_vectors(
        embedder.name, [(sha, no, off, vec) for (sha, no, off), vec in zip(rows, vecs)]
    )
    return len(todo)


def update_index(
    store: MemoryStore, index: SearchIndex, session: str, embedder=None
) -> tuple[int, int, int]:
    """Index a session's items beyond what the index already has."""
    done = index.max_seq(session)
    rows = []
    for line in store.git("ls-tree", "-r", f"refs/heads/{session}").splitlines():
        meta, _, path = line.partition("\t")
        m = ITEM_RE.match(path)
        if m and int(m.group(1)) > done:
            rows.append((int(m.group(1)), m.group(2), m.group(3), meta.split()[2]))
    if not rows:
        return 0, 0, 0
    rows.sort()
    fresh = index.missing_blobs({sha for _, _, _, sha in rows})
    contents = store.cat_batch(sorted(fresh))
    index.add(session, rows, contents)
    embedded = embed_blobs(index, embedder, contents) if embedder and contents else 0
    return len(rows), len(fresh), embedded


def ingest_all(
    store: MemoryStore,
    index: SearchIndex,
    root: Path = DEFAULT_ROOT,
    jobs: int = 1,
    force: bool = False,
    embedder=None,
    progress=lambda msg: None,
) -> IngestStats:
    t0 = time.time()
    stats = IngestStats()
    watermark = float(index.get_meta(MTIME_KEY) or 0)
    files = [f for f in discover(root) if force or f.stat().st_mtime > watermark]
    stats.files_seen = len(files)
    if not files:
        stats.seconds = time.time() - t0
        return stats
    max_mtime = max(f.stat().st_mtime for f in files)

    if jobs > 1 and len(files) > 1:
        with Pool(jobs) as pool:
            results = pool.imap_unordered(
                _worker, [(str(store.path), str(f), str(root)) for f in files]
            )
            for i, (session, appended) in enumerate(results, 1):
                if appended:
                    stats.files_ingested += 1
                    stats.items_appended += appended
                    stats.sessions.append(session)
                progress(f"[{i}/{len(files)}] {stats.items_appended:,} items appended")
    else:
        for i, f in enumerate(files, 1):
            session, appended = ingest_file(store, f, root)
            if appended:
                stats.files_ingested += 1
                stats.items_appended += appended
                stats.sessions.append(session)
            progress(f"[{i}/{len(files)}] {stats.items_appended:,} items appended")

    for session in stats.sessions:
        n_items, n_blobs, n_embedded = update_index(store, index, session, embedder)
        stats.items_indexed += n_items
        stats.blobs_indexed += n_blobs
        stats.blobs_embedded += n_embedded
    index.set_meta(MTIME_KEY, repr(max_mtime))
    stats.seconds = time.time() - t0
    return stats
