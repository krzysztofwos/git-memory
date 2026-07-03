"""Micro-benchmark: is Git viable as the hot path of an agent loop?

Measures, for a synthetic-but-realistic transcript (short messages, multi-KB
tool results, 25% of tool results are exact re-reads):

  - per-append latency (this prototype shells out to git plumbing; a libgit2
    binding would cut this ~10x -- the number here is the *ceiling*)
  - context materialization latency at tip and at a historical commit
  - storage: git repo (every context state kept) vs a JSONL event log
    (no state history) vs naive one-snapshot-per-state (what keeping every
    state costs without structural sharing)

Run: python3 bench.py [N_EVENTS]
"""

import gzip
import json
import random
import statistics
import sys
import time

from gitmem import MemoryStore

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = random.Random(42)

WORDS = ("payment retry backoff worker charge idempotency jitter queue "
         "config deploy trace error timeout socket parse token cache").split()


def prose(n_chars: int) -> str:
    out = []
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(rng.choice(WORDS))
    return " ".join(out)


# Build event stream: repeating (user, assistant, tool_call, tool_result) turns.
tool_results: list[str] = []
events: list[tuple[str, str, str]] = [("message", "system", prose(1500))]
while len(events) < N:
    events.append(("message", "user", prose(rng.randint(100, 400))))
    events.append(("message", "assistant", prose(rng.randint(200, 800))))
    events.append(("tool_call", "assistant", f'read("src/file{rng.randint(1, 40)}.py")'))
    if tool_results and rng.random() < 0.25:
        tr = rng.choice(tool_results)  # re-read of an unchanged file
    else:
        tr = prose(rng.randint(2000, 6000))
        tool_results.append(tr)
    events.append(("tool_result", "tool", tr))
events = events[:N]

store = MemoryStore("stores/bench.git", fresh=True)
sess = store.session("main")

lat = []
jsonl_lines = []
state_bytes = 0  # cumulative: size of every context state if snapshotted naively
running_state = 0
for kind, role, content in events:
    t0 = time.perf_counter()
    sess.append(kind, role, content)
    lat.append((time.perf_counter() - t0) * 1000)
    line = json.dumps({"kind": kind, "role": role, "content": content})
    jsonl_lines.append(line)
    running_state += len(line) + 1
    state_bytes += running_state

jsonl_raw = "\n".join(jsonl_lines).encode() + b"\n"
jsonl_bytes = len(jsonl_raw)
jsonl_gz = len(gzip.compress(jsonl_raw))

def pct(p):
    return statistics.quantiles(lat, n=100)[p - 1]

mid = store.git("rev-list", "main").splitlines()[N // 2]
t0 = time.perf_counter(); items = sess.materialize(); mat_tip = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter(); old = sess.materialize(mid); mat_mid = (time.perf_counter() - t0) * 1000

size_pre = store.size_bytes()
store.gc()
size_post = store.size_bytes()

print(f"events appended            {N}  (final context: {len(items)} items, "
      f"~{sess.token_total():,} tok)")
print(f"append latency ms          mean {statistics.mean(lat):.1f}  "
      f"p50 {statistics.median(lat):.1f}  p95 {pct(95):.1f}  max {max(lat):.1f}")
print(f"materialize tip / mid ms   {mat_tip:.1f} / {mat_mid:.1f}  "
      f"({len(items)} / {len(old)} items)")
print()
print(f"JSONL event log (no state history)      {jsonl_bytes:>12,} bytes")
print(f"JSONL gzipped (no state history)        {jsonl_gz:>12,} bytes")
print(f"naive snapshot-per-state ({N} states)   {state_bytes:>12,} bytes")
print(f"git repo, every state, pre-gc           {size_pre:>12,} bytes")
print(f"git repo, every state, post-gc          {size_post:>12,} bytes  "
      f"({size_post / jsonl_bytes:.2f}x JSONL, "
      f"{size_post / state_bytes:.4f}x naive snapshots)")
