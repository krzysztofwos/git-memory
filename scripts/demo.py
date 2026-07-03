"""Narrative demo: a simulated agent session stored entirely in Git.

Exercises every capability the design claims:
  1. every context event is a commit; the prompt is a pure function of a SHA
  2. compaction is a commit, not a rewrite -- auditable via `git diff`
  3. a fact compacted out of the live context is recoverable from history
  4. subagents are branches; absorbing one is a merge commit (provenance)
  5. time travel: reconstruct the exact context of any earlier LLM call

Run: uv run scripts/demo.py
"""

from gitmem import MemoryStore

FILE_READ = """\
# payments/retry.py
import time

MAX_RETRY_BACKOFF_SECONDS = 3600  # undocumented cap added by ops, 2024-11
RETRY_BASE = 2

def backoff(attempt: int) -> int:
    return min(RETRY_BASE ** attempt, MAX_RETRY_BACKOFF_SECONDS)

def retry(fn, attempts=8):
    for i in range(attempts):
        try:
            return fn()
        except TransientError:
            time.sleep(backoff(i))
    raise
"""

GREP_RESULT = """\
payments/retry.py:4:MAX_RETRY_BACKOFF_SECONDS = 3600
payments/worker.py:12:from .retry import retry
payments/worker.py:88:    retry(charge, attempts=8)
"""


def h(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


store = MemoryStore("stores/demo.git", fresh=True)
main = store.session("main")

h("1. An agent session unfolds; every event is a commit")
main.append("message", "system", "You are a coding agent. Investigate and fix bugs.")
main.append("message", "user", "Payment retries hammer the API. Investigate payments/.")
main.append("tool_call", "assistant", 'grep("MAX_RETRY", "payments/")')
main.append("tool_result", "tool", GREP_RESULT)
main.append("tool_call", "assistant", 'read("payments/retry.py")')
main.append("tool_result", "tool", FILE_READ)
main.append(
    "message",
    "assistant",
    "Backoff is capped at MAX_RETRY_BACKOFF_SECONDS=3600 (an undocumented ops "
    "cap from 2024-11), and worker.py retries 8 times. The hammering comes "
    "from worker.py's tight attempts, not the cap. Investigating worker.py next.",
)
main.append("tool_call", "assistant", 'read("payments/retry.py")')
main.append("tool_result", "tool", FILE_READ)  # re-read: dedup, same blob
print(main.log_oneline())
items = main.materialize()
dupes = [i for i in items if i.blob == items[5].blob]
print(f"Re-read file stored once: items {[i.seq for i in dupes]} share blob "
      f"{items[5].blob[:12]}")
print(f"Context: {len(items)} items, ~{main.token_total()} tokens")

h("2. Compaction is a commit -- the diff IS the audit trail")
pre_compact_tip = main.tip
main.compact(
    2, 9,
    "Investigated payment retry hammering. Grep + read of payments/retry.py: "
    "exponential backoff capped by an ops-added constant; worker.py retries "
    "charge() 8 times. Root cause suspected in worker.py retry loop.",
)
print(store.git("log", "--oneline", "-1", "main"))
print(f"\nLive context now: {[(i.seq, i.kind) for i in main.materialize()]}")
print(f"Tokens: ~{main.token_total()}")
print("\n`git diff` of the compaction (what did the summarizer drop?):")
diff = store.git("diff", "--stat", f"{main.tip}~1", main.tip)
print(diff)

h("3. The needle: a fact now absent from live context is recoverable")
needle = "MAX_RETRY_BACKOFF_SECONDS = 3600"
live = main.prompt_text()
print(f"In live prompt after compaction? {needle in live}")
hits = store.grep_history(r"MAX_RETRY_BACKOFF_SECONDS = \d+")
for hit in hits:
    print(f"History hit: {hit.path} (blob {hit.blob[:12]}): {hit.snippet.strip()}")
recovered = store.retrieve(hits[0].blob)
print(f"Full original recovered by content address? {needle in recovered}")

h("4. Subagent = branch; absorbing it = merge commit with provenance")
sub = main.fork("sub-worker-audit")
sub.append("message", "user", "Subtask: audit payments/worker.py retry loop.")
sub.append("tool_call", "assistant", 'read("payments/worker.py")')
sub.append("tool_result", "tool", "...worker.py source: retry(charge, attempts=8) "
           "with no jitter and no idempotency key...")
main.absorb(sub, "Subagent verdict: worker.py retries charge() 8x with no "
                 "jitter/idempotency key -- that is the API hammering. Fix: add "
                 "jitter + idempotency key, drop attempts to 4.")
print(store.git("log", "--graph", "--oneline", "--all"))
print(f"Main context items: {[(i.seq, i.kind) for i in main.materialize()]}")

h("5. Time travel: rebuild the exact prompt of any earlier LLM call")
old = main.materialize(pre_compact_tip)
now = main.materialize()
print(f"Context at pre-compaction commit {pre_compact_tip[:12]}: "
      f"{len(old)} items, needle present: {any(needle in i.content for i in old)}")
print(f"Context at tip: {len(now)} items, needle present: "
      f"{any(needle in i.content for i in now)}")

h("Storage")
store.gc()
print(f"Repo size after gc: {store.size_bytes():,} bytes for "
      f"{len(store.git('rev-list', '--all').splitlines())} context states")
