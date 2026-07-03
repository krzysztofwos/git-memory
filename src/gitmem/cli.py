"""gitmem CLI: a searchable, verbatim archive of every Claude Code session.

Layout under ~/.claude/gitmem (override with GITMEM_HOME or --home):
    store.git      bare git store, one branch per session
    index.sqlite   FTS5 index keyed by blob SHA
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from gitmem.core import GitError, MemoryStore
from gitmem.index import SearchIndex
from gitmem.ingest import ingest_all
from gitmem.transcript import DEFAULT_ROOT


def default_home() -> Path:
    return Path(os.environ.get("GITMEM_HOME", Path.home() / ".claude/gitmem"))


def open_all(home: Path) -> tuple[MemoryStore, SearchIndex]:
    home.mkdir(parents=True, exist_ok=True)
    return MemoryStore(home / "store.git"), SearchIndex(home / "index.sqlite")


def cmd_ingest(args) -> int:
    import fcntl

    args.home.mkdir(parents=True, exist_ok=True)
    lock = open(args.home / ".ingest.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not args.quiet:
            print("another ingest is running; nothing to do")
        return 0
    store, index = open_all(args.home)
    progress = (lambda m: None) if args.quiet else (
        lambda m: print(f"\r  {m}", end="", flush=True)
    )
    embedder = None
    if not args.no_embed:
        from gitmem.embed import load_embedder

        embedder = load_embedder()
    stats = ingest_all(
        store, index, root=args.projects, jobs=args.jobs, force=args.force,
        embedder=embedder, progress=progress,
    )
    if not args.quiet:
        print()
    if stats.items_appended or not args.quiet:
        print(
            f"ingested {stats.items_appended:,} new items from "
            f"{stats.files_ingested} sessions ({stats.blobs_indexed:,} new blobs "
            f"indexed, {stats.blobs_embedded:,} embedded) in {stats.seconds:.1f}s"
        )
    return 0


def cmd_embed(args) -> int:
    """Backfill vectors for already-indexed blobs missing them."""
    from gitmem.embed import MODEL_NAME, load_embedder

    from gitmem.ingest import embed_blobs

    store, index = open_all(args.home)
    embedder = load_embedder()
    if embedder is None:
        print("embedding model unavailable (is fastembed installed?)", file=sys.stderr)
        return 1
    todo = sorted(index.missing_vectors(
        {r[0] for r in index.db.execute("SELECT sha FROM blobs")}, MODEL_NAME,
    ))
    print(f"{len(todo):,} blobs to embed")
    done = 0
    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        contents = store.cat_batch(batch)
        done += embed_blobs(index, embedder, contents)
        print(f"\r  {done:,}/{len(todo):,} blobs "
              f"({index.vector_count(MODEL_NAME):,} vectors)", end="", flush=True)
    print()
    return 0


def cmd_search(args) -> int:
    from gitmem.embed import MODEL_NAME

    store, index = open_all(args.home)
    query = " ".join(args.query)
    qvec = None
    if not args.exact and index.vector_count(MODEL_NAME):
        from gitmem.embed import load_embedder

        embedder = load_embedder()
        if embedder is not None:
            qvec = embedder.embed_query(query)
    if args.semantic and qvec is None:
        print("semantic search unavailable (no vectors or no model)", file=sys.stderr)
        return 1
    if args.semantic:
        ranked = index.semantic_ranked(qvec, MODEL_NAME, args.limit * 4)
        hits = _semantic_hits(index, ranked, args)
    else:
        hits = index.hybrid_search(
            query, qvec, MODEL_NAME, limit=args.limit,
            kind=args.kind, session_like=args.session,
        )
    if not hits:
        print("no matches")
        return 1
    for h in hits:
        session, seq, kind = h.occurrences[0]
        more = f"  (+{h.n_occurrences - 1} more places)" if h.n_occurrences > 1 else ""
        print(f"● [{h.origin}] {kind}  {session} #{seq}  blob {h.sha[:12]}{more}")
        snippet = h.snippet or _excerpt(store, h.sha, h.offset)
        print(f"  {' '.join(snippet.split())}\n")
    print("full content: gitmem show <blob> | context: gitmem timeline <session> <seq>")
    return 0


def _semantic_hits(index, ranked, args):
    hits = []
    for sha, off, score in ranked:
        occ = index._occurrences(sha, args.kind, args.session)
        if not occ:
            continue
        from gitmem.index import SearchHit

        hits.append(SearchHit(sha, None, score, occ[:3], len(occ), "sem", off))
        if len(hits) >= args.limit:
            break
    return hits


def _excerpt(store, sha, offset, width=200):
    content = store.retrieve(sha)
    return content[offset : offset + width]


def cmd_show(args) -> int:
    store, _ = open_all(args.home)
    try:
        sys.stdout.write(store.retrieve(args.sha))
    except GitError:
        print(f"no blob {args.sha}", file=sys.stderr)
        return 1
    return 0


def resolve_session(store: MemoryStore, needle: str) -> str:
    names = [
        line.strip() for line in
        store.git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
    ]
    exact = [n for n in names if n == needle]
    matches = exact or [n for n in names if needle in n]
    if len(matches) != 1:
        raise SystemExit(
            f"session '{needle}' matches {len(matches)} branches"
            + (f": {', '.join(matches[:5])}..." if matches else "")
        )
    return matches[0]


def cmd_timeline(args) -> int:
    store, _ = open_all(args.home)
    session = resolve_session(store, args.session)
    items = store.session(session).materialize()
    window = [i for i in items if abs(i.seq - args.seq) <= args.context]
    for it in window:
        marker = ">>" if it.seq == args.seq else "  "
        body = it.content if args.full else " ".join(it.content[:300].split())
        print(f"{marker} {it.header()} (blob {it.blob[:12]})\n   {body}\n")
    return 0 if window else 1


def cmd_sessions(args) -> int:
    store, _ = open_all(args.home)
    out = store.git(
        "for-each-ref", "--sort=-committerdate", f"--count={args.limit}",
        "--format=%(committerdate:short)  %(refname:short)\n"
        "            %(subject)", "refs/heads",
    )
    print(out.rstrip() or "empty store — run: gitmem ingest")
    return 0


def cmd_gc(args) -> int:
    # Default git-gc under-packs this workload ~3x (near-identical trees need
    # a large delta window) — measured in the H2 replay; see README.
    store, _ = open_all(args.home)
    before = store.size_bytes()
    store.git("repack", "-adf", "--window=250", "--depth=50", "--threads=0")
    store.git("prune-packed")
    print(f"repacked: {before / 1e6:,.1f} MB -> {store.size_bytes() / 1e6:,.1f} MB")
    return 0


def cmd_stats(args) -> int:
    store, index = open_all(args.home)
    c = index.counts()
    branches = len(store.git("for-each-ref", "refs/heads").splitlines())
    print(f"store    {store.path}  ({store.size_bytes() / 1e6:,.1f} MB)")
    print(f"sessions {branches}")
    from gitmem.embed import MODEL_NAME

    print(f"indexed  {c['items']:,} items, {c['blobs']:,} unique blobs, "
          f"{c['sessions']} sessions")
    print(f"vectors  {index.vector_count(MODEL_NAME):,} ({MODEL_NAME})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gitmem", description="Searchable verbatim archive of Claude Code sessions"
    )
    ap.add_argument("--home", type=Path, default=default_home(),
                    help="archive location (default: ~/.claude/gitmem)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="ingest new/changed transcripts (incremental)")
    p.add_argument("--projects", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true", help="rescan all files, ignore mtime watermark")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding stage")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("embed", help="backfill vectors for indexed blobs")
    p.add_argument("--batch", type=int, default=1024)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("search", help="full-text search across all sessions")
    p.add_argument("query", nargs="+")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("--kind", help="filter: message|tool_call|tool_result|thinking")
    p.add_argument("--session", help="filter: substring of session name")
    p.add_argument("--exact", action="store_true", help="FTS only (fast, no model load)")
    p.add_argument("--semantic", action="store_true", help="vectors only")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("show", help="print a blob's verbatim content")
    p.add_argument("sha")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("timeline", help="show items around a position in a session")
    p.add_argument("session", help="session name or unique substring")
    p.add_argument("seq", type=int)
    p.add_argument("-n", "--context", type=int, default=3)
    p.add_argument("--full", action="store_true", help="don't truncate item content")
    p.set_defaults(fn=cmd_timeline)

    p = sub.add_parser("sessions", help="list sessions, most recent first")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.set_defaults(fn=cmd_sessions)

    p = sub.add_parser("stats", help="store and index statistics")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("gc", help="repack the store with tuned delta window")
    p.set_defaults(fn=cmd_gc)

    p = sub.add_parser("setup", help="install Claude Code SessionStart hook and skill")
    p.add_argument("--claude-dir", type=Path, default=Path.home() / ".claude")
    p.set_defaults(fn=None)  # bound lazily to avoid import cost on every command

    args = ap.parse_args(argv)
    if args.cmd == "setup":
        from gitmem.setup import cmd_setup
        return cmd_setup(args)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
