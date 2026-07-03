"""Replay real Claude Code transcripts through gitmem to test H2.

H2: a git store holding *every historical context state* costs no more than
the raw event log that holds none of them.

Every ~/.claude/projects/**/*.jsonl transcript becomes one branch in a single
shared store (stores/replay.git). Each context-bearing block (user/assistant
text, thinking, tool_use, tool_result) becomes one Session.append() -> one
commit. Replay is append-only: the transcript is the event log, so no
compaction ops are injected. Workers run in parallel across transcripts --
loose-object writes are atomic and branches don't contend, so this doubles
as the concurrency test.

Reported baselines:
  - raw JSONL bytes (includes uuids/timestamps/etc. that gitmem never stores)
  - extracted content bytes: just the item contents, i.e. a minimal event log
    -- the *fair* baseline for H2
  - gzip of extracted content (per-session archives, summed)
  - naive snapshot-per-state (computed arithmetically, not written)

Run: uv run scripts/replay_h2.py [--jobs N] [--limit K]
"""

import argparse
import gzip
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

from gitmem import MemoryStore

STORE = "stores/replay.git"
ROOT = Path.home() / ".claude/projects"

KIND_RE = re.compile(r"[^a-z_]")


def flatten(content) -> str:
    """tool_result content: string, or list of text/image/... blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            else:
                parts.append(json.dumps(b, sort_keys=True, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def extract_events(path: Path) -> list[tuple[str, str, str]]:
    """Transcript line -> the context items it contributes, in order."""
    events = []

    def add(kind, role, content):
        if content:
            events.append((KIND_RE.sub("_", kind.lower()) or "other", role, content))

    for line in open(path, errors="replace"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = rec.get("type")
        if t not in ("user", "assistant"):
            continue
        content = (rec.get("message") or {}).get("content")
        if isinstance(content, str):
            role = "user" if t == "user" else "assistant"
            add("message", role, content)
            continue
        for b in content or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                add("message", "user" if t == "user" else "assistant", b.get("text", ""))
            elif bt == "thinking":
                add("thinking", "assistant", b.get("thinking", ""))
            elif bt == "tool_use":
                add("tool_call", "assistant", json.dumps(
                    {"name": b.get("name"), "input": b.get("input")},
                    sort_keys=True, ensure_ascii=False))
            elif bt == "tool_result":
                add("tool_result", "tool", flatten(b.get("content")))
            else:
                add(bt or "other", "user" if t == "user" else "assistant",
                    json.dumps(b, sort_keys=True, ensure_ascii=False))
    return events


def slug(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(rel)).strip("-")
    return s[-100:].strip("-")  # keep the unique uuid tail


def replay_one(path_str: str) -> dict:
    path = Path(path_str)
    stats = {"file": path.name, "raw": path.stat().st_size, "items": 0,
             "content": 0, "gz": 0, "naive": 0, "secs": 0.0, "skipped": False}
    events = extract_events(path)
    if not events:
        return stats
    store = MemoryStore(STORE)
    sess = store.session(slug(path))
    if sess.tip is not None:  # already replayed (resumable reruns)
        stats["skipped"] = True
        return stats
    buf, running = [], 0
    t0 = time.time()
    for kind, role, content in events:
        sess.append(kind, role, content)
        nbytes = len(content.encode())
        stats["items"] += 1
        stats["content"] += nbytes
        running += nbytes
        stats["naive"] += running
        buf.append(content)
    stats["secs"] = time.time() - t0
    stats["gz"] = len(gzip.compress("\n".join(buf).encode()))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="only the K largest files")
    args = ap.parse_args()

    files = sorted(ROOT.rglob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    if args.limit:
        files = files[: args.limit]
    store = MemoryStore(STORE)  # init once before forking

    totals = {"raw": 0, "items": 0, "content": 0, "gz": 0, "naive": 0}
    done = skipped = 0
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for st in pool.imap_unordered(replay_one, map(str, files)):
            done += 1
            skipped += st["skipped"]
            for k in totals:
                totals[k] += st[k]
            if st["items"] >= 2000:
                print(f"  [{done}/{len(files)}] {st['file']}: {st['items']:,} items "
                      f"in {st['secs']:.0f}s ({st['secs'] / st['items'] * 1000:.0f} ms/append)")
            elif done % 50 == 0:
                print(f"  [{done}/{len(files)}] ... {totals['items']:,} items so far")
    wall = time.time() - t0

    print(f"\nreplayed {done - skipped} transcripts ({skipped} skipped), "
          f"{totals['items']:,} items, wall {wall / 60:.1f} min")

    size_pre = store.size_bytes()
    t0 = time.time()
    store.git("gc", "--quiet")
    size_post = store.size_bytes()
    print(f"gc took {time.time() - t0:.0f}s")

    counts: dict[str, tuple[int, int]] = {}
    for line in store.git_bytes(
        "cat-file", "--batch-all-objects", "--batch-check=%(objecttype) %(objectsize)"
    ).decode().splitlines():
        typ, size = line.split()
        n, s = counts.get(typ, (0, 0))
        counts[typ] = (n + 1, s + int(size))

    states = sum(1 for _ in store.git("rev-list", "--all").splitlines())
    mb = 1e6
    print(f"\n=== H2 on real transcripts ===")
    print(f"context items appended                  {totals['items']:>14,}")
    print(f"context states stored (commits)         {states:>14,}")
    for typ in ("blob", "tree", "commit"):
        n, s = counts.get(typ, (0, 0))
        print(f"  unique {typ:6}                        {n:>10,}  ({s / mb:,.1f} MB uncompressed)")
    print()
    print(f"raw JSONL transcripts                   {totals['raw'] / mb:>11,.1f} MB")
    print(f"extracted content (minimal event log)   {totals['content'] / mb:>11,.1f} MB")
    print(f"gzipped content (per-session, summed)   {totals['gz'] / mb:>11,.1f} MB")
    print(f"naive snapshot-per-state                {totals['naive'] / mb:>11,.1f} MB")
    print(f"git store, ALL states, pre-gc           {size_pre / mb:>11,.1f} MB")
    print(f"git store, ALL states, post-gc          {size_post / mb:>11,.1f} MB")
    print(f"\ngit/raw-JSONL      = {size_post / totals['raw']:.3f}x")
    print(f"git/minimal-log    = {size_post / totals['content']:.3f}x   <- H2: must be <= 1")
    print(f"git/gzipped        = {size_post / totals['gz']:.3f}x")
    print(f"git/naive-states   = {size_post / totals['naive']:.5f}x")
    dedup = counts.get("blob", (0, 0))[1]
    print(f"dedup: {totals['items']:,} items -> {counts.get('blob', (0, 0))[0]:,} unique blobs "
          f"({totals['content'] / mb:,.1f} MB referenced, {dedup / mb:,.1f} MB unique)")


if __name__ == "__main__":
    main()
