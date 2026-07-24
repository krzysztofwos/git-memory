# git-memory

An experiment: use Git's object model — not just a Git repo of files — as the
storage substrate for LLM agent memory. Every piece of context (messages, tool
calls, tool results) **and** every operation applied to it (compaction,
subagent fork/merge, context edits) is recorded in one content-addressed DAG.

## The core invariant

> **The agent's context window is a pure function of a commit SHA.**

The tree at HEAD _is_ the context. To build the prompt for the next LLM call,
materialize the tree at the session's tip. Every mutation is a new commit:

| Agent concept                                    | Git primitive                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| Context item (message / tool call / tool result) | blob (raw content only)                                            |
| Context state at any point in time               | tree                                                               |
| Operation (append, compact, edit)                | commit (op metadata in message trailers)                           |
| Session                                          | branch                                                             |
| Subagent spawn                                   | branch fork at parent's tip                                        |
| Subagent result returned to parent               | merge commit (2nd parent = subagent tip)                           |
| Checkpoint / milestone                           | tag                                                                |
| Operation log                                    | `git log`                                                          |
| "What did compaction drop?"                      | `git diff`                                                         |
| Cross-session / long-term memory                 | other branches (retrieval = read + append with provenance trailer) |
| Post-hoc annotations (eval scores, embeddings)   | `git notes` (no history rewrite)                                   |

**Compaction is a commit, never a rewrite.** The compaction commit's tree
replaces N item blobs with one summary blob. The pre-compaction state remains
reachable as the parent commit, so compaction becomes _recoverable lossy
compression_: the live context shrinks, but any fact that was ever in context
can be found again (`grep_history`) and retrieved verbatim by its blob SHA.

## Layout

Item metadata lives in the filename. Operation metadata lives in commit
trailers. A blob is _exactly_ the raw content. Items are fanned out into
buckets of 256 so an append rewrites one small bucket subtree plus the root
— a flat directory makes total tree bytes quadratic in session length, which
real 10k-item sessions punished badly (see the replay results below):

```text
items/0000/000001.message.system.md
items/0000/000002.summary.assistant.md   # compaction reused seq 2's slot
items/0000/000010.summary.assistant.md   # absorbed subagent result
items/0003/000841.tool_result.tool.md
state.json                               # {"next_seq": N}
```

Keeping blobs metadata-free is what makes content addressing work as
deduplication: an agent re-reading an unchanged file produces the same blob
SHA, stored once no matter how many context states reference it.

```text
Op: compact                 # commit trailers = machine-readable op log
Replaced: 2..9
Items-Removed: 8
Tokens-Total: 63
```

## What Git gives you for free

- **Time travel / replay**: reconstruct the exact prompt of any past LLM call
  from its commit SHA — deterministic debugging of agent runs.
- **Auditable compaction**: `git diff` on a compaction commit shows precisely
  what the summary elided. Reviewable in any Git UI, even a PR review tool.
- **Deduplication**: repeated tool results, repeated system prompts across
  sessions — one blob. Packfile delta compression handles near-duplicates
  (file read before/after a small edit).
- **Branching**: counterfactuals ("fork at turn 12, vary one tool result"),
  best-of-n rollouts, subagent contexts with full provenance of fork point
  and merge point.
- **Retrieval over everything ever stored**: `git grep`/pickaxe across
  history reaches content compacted out of every live context.
- **Tamper-evident audit log**: the hash chain. Add signed commits for
  attestation.
- **Concurrency**: `update-ref` with an expected old value is compare-and-swap.
  Sessions on separate branches don't contend.
- **An embedding cache key**: blob SHAs are immutable content addresses —
  embed each blob once, never invalidate.

## Package

Zero-dependency Python over git plumbing (`hash-object`, `mktree`,
`commit-tree`, `update-ref`) against a bare repo. uv project. `uv sync` to
set up, `uv run pytest` to test.

- `src/gitmem/core.py` — the store (`MemoryStore`, `Session`: append /
  compact / fork / absorb / materialize / grep_history / retrieve)
- `src/gitmem/transcript.py` — Claude Code JSONL → context events
- `src/gitmem/index.py` — SQLite FTS5 + vector index keyed by blob SHA (a
  blob is indexed and embedded once, ever), hybrid search via
  reciprocal-rank fusion
- `src/gitmem/embed.py` — local embedding model (bge-small via fastembed,
  ONNX/CPU, no torch) and the deterministic chunker
- `src/gitmem/ingest.py` — incremental ingestion, resumable mid-session at
  the item level
- `src/gitmem/cli.py`, `src/gitmem/setup.py` — the `gitmem` CLI and the
  Claude Code integration installer
- `scripts/demo.py` — scripted agent session exercising every claim above,
  including recovering a compacted-away fact
- `scripts/bench.py` — latency + storage micro-benchmark on synthetic events
- `scripts/replay_h2.py` — reproduces the H2 measurement below

```sh
uv run scripts/demo.py
uv run scripts/bench.py 1000
uv run scripts/replay_h2.py --jobs 12
```

## Using it with Claude Code

The archive lives in `~/.claude/gitmem/` (`store.git` + `index.sqlite`),
mirroring `~/.claude/projects/**/*.jsonl` — one branch per session,
subagent transcripts included.

```sh
uv run gitmem ingest --jobs 12   # initial ingest (minutes). Later runs are incremental
uv run gitmem embed              # one-time vector backfill (downloads model on first use)
uv run gitmem setup              # install SessionStart hook + gitmem skill

uv run gitmem search "connection pool exhausted"     # hybrid: FTS + vectors, RRF-fused
uv run gitmem search --exact "MAX_RETRY_BACKOFF"     # FTS only (fast, no model load)
uv run gitmem search --semantic "that weird build issue with linking"
uv run gitmem search --kind tool_result --session theseus "traceback"
uv run gitmem show <blob-sha>            # verbatim original
uv run gitmem timeline <session> <seq>   # what surrounded a hit
uv run gitmem sessions
uv run gitmem stats
```

Search is hybrid by default: bm25 over FTS5 plus cosine over locally-computed
embeddings (bge-small, ONNX on CPU — nothing leaves the machine), fused by
reciprocal rank. Exact identifiers win via the FTS leg. Vague paraphrases win
via the vector leg. Embeddings inherit the blob-SHA property: computed once
per unique content, no invalidation path, incremental by set-difference. At
this scale (~100k vectors) similarity is brute-force exact — no ANN index to
maintain.

`setup` writes `~/.claude/skills/gitmem/SKILL.md` (teaches Claude when to
search the archive) and adds a `SessionStart` hook running
`gitmem ingest --quiet`, so the archive refreshes itself at session start
(sub-second once initialized, thanks to the mtime watermark + item-level
resume). Both reference this checkout's `.venv/bin/gitmem` by absolute
path. Re-run `gitmem setup` if the project moves.

Privacy note: the archive contains everything that ever passed through a
session, including secrets in old tool output. It is local-only. Treat it
with the same care as `~/.claude/projects` itself.

### Synthetic benchmark (git 2.43, Python 3.12, subprocess plumbing)

1000 events (~290k tokens final context, 25% duplicate tool results):
append 15.7 ms mean / 19 ms p95, flat w.r.t. context size; materialize a
1000-item context ~43 ms; repo with all 1000 states post-gc 926 KB vs
1,220 KB raw JSONL and 607 MB naive per-state snapshots. The ~16 ms append
is a subprocess-spawn ceiling (6–7 git processes per commit).
libgit2/pygit2 does the same object writes in well under a millisecond.
Either way it is noise against multi-second LLM calls.

### Real-data replay: the H2 experiment

334 real Claude Code transcripts (311 MB JSONL, 71,399 context items,
largest session 10,102 items) replayed through `Session.append`, one branch
per session, 12 workers sharing one object store:

| storage of the same corpus                       | size        | vs minimal event log |
| ------------------------------------------------ | ----------- | -------------------- |
| raw JSONL transcripts                            | 311.3 MB    | 3.56x                |
| extracted content (minimal event log, no states) | 87.4 MB     | 1.00x                |
| gzipped content (per-session archives)           | 20.6 MB     | 0.24x                |
| naive snapshot-per-state (computed)              | 114.2 GB    | 1306x                |
| git **flat** layout, default `git gc`            | 294.3 MB    | 3.37x                |
| git flat, `repack --window=250`                  | 89.9 MB     | 1.03x                |
| git **fanout** layout, default `git gc`          | 168.7 MB    | 1.93x                |
| git fanout, `repack --window=250`                | **90.2 MB** | **1.03x**            |

**Verdict: H2 as stated (≤ 1.0x) is narrowly falsified — the true number is
1.03x.** Every one of the 71k context states plus full operation provenance
costs 3% more than an event log holding none of them, and 0.29x what Claude
Code's own JSONL takes today. The spirit holds. The letter missed by 3%.

What the experiment actually taught:

1. **Flat trees are a trap.** The first run produced 7.3 GB of uncompressed
   tree objects (quadratic in session length) — 3.4x the minimal log after
   default gc, with appends degrading to 88 ms on the 10k-item session. The
   synthetic benchmark at 1000 events never showed this. Fanout cut
   uncompressed tree bytes 17x (to 435 MB), kept appends flat at 17 ms, and
   cut replay wall time 5x (14.9 → 2.9 min).
2. **Pack window matters more than layout for cold size.** With
   `--window=250`, flat and fanout converge to the same 90 MB: near-identical
   trees delta away almost entirely. Default gc under-packs this workload
   (1.9–3.4x). A memory store should tune repack settings.
3. **The residual cost is commit/tree metadata, not content.** Final pack:
   blobs 26.3 MB, trees 31.2 MB, commits 14.9 MB compressed. Provenance
   machinery is ~1.8x the deduped content — the lever for getting below
   1.0x is fewer, batched commits (per assistant turn instead of per
   content block, ~3–4x fewer states), not better trees.
4. **Exact-duplicate dedup on real data is 12%** (87.4 MB referenced →
   77.1 MB unique blobs), less than the synthetic 25%: real duplicates are
   near-duplicates (timestamps, ids inside tool results), which pack deltas
   catch instead.
5. **Concurrency held.** 12 parallel writers into one object store, 334
   branches, zero conflicts (atomic loose-object writes + per-branch CAS
   ref updates).

Gzip remains 4.4x smaller — the honest gap: git is a _queryable, replayable_
store at 1.03x the log, not a minimal archive at 0.24x.

Read path on real data: materializing the largest session's full context
(10,102 items, ~2.3M est. tokens) from its SHA takes 146 ms. A mid-history
state (5,051 items) takes 75 ms.

Stores left in place for inspection: `stores/replay.git` (fanout) and
`stores/replay-flat.git` (flat) — browse with any git tooling, e.g.
`git -C stores/replay.git log --oneline <branch>`.

## Known limitations

- **Granularity**: commit per event, not per streaming token.
- **No semantic retrieval**: grep/pickaxe only. Bolt on an embedding index
  keyed by blob SHA (immutable → embed once).
- **Large binary tool results** (screenshots): work, but cap sizes or use
  LFS-style pointer blobs past a threshold.
- **Discipline required**: the model's guarantees hold only if history is
  never rewritten and refs to abandoned forks are kept (nothing becomes
  unreachable → `git gc` prunes nothing).
- **Token counts are estimates** (len/4 here). Real accounting needs the
  target model's tokenizer and lives in trailers/notes, not blobs.
- **One writer per branch**: CAS on `update-ref` makes violations loud, not
  impossible.

## Evaluation plan

Two axes: does Git _hold up_ as a store, and does this memory model _help the
agent_?

### Systems (cheap, falsifiable now)

- **H1: append latency < 25 ms p95 at 10^3 commits, flat in context size.**
  Confirmed — 19 ms p95 synthetic, 17 ms mean sustained across a 10k-item
  real session with the fanout layout (flat degraded to 88 ms). Next:
  10^5–10^6 commits, pygit2 (< 1 ms).
- **H2: storing all context states costs ≤ raw event log bytes** on real
  transcripts. **Run — narrowly falsified: 1.03x** the minimal extracted
  event log (see the replay section above for the full result and the
  flat-vs-fanout/pack-window findings). Batching commits per assistant turn
  is the identified path below 1.0x.
- Reconstruction latency at tip vs. arbitrary historical commit vs. JSONL
  replay. Concurrency: confirmed with 12 parallel writers / 334 branches in
  the replay.

### Agent quality (the interesting part)

1. **Recoverable vs. lossy compaction** (headline experiment). Long-horizon
   tasks with facts planted early, three arms:
   (a) truncation, (b) summary-only compaction, (c) git-memory compaction
   plus a `memory_search` tool that greps history and retrieves original
   blobs. Measure fact recall, task success, tokens consumed. Hypothesis:
   (c) recovers most planted facts, while (a)/(b) approach zero as distance
   grows.
2. **Compaction faithfulness audits.** Every compaction is a diff, so mine
   (dropped content, summary) pairs from real runs and score information
   loss with an LLM judge — a benchmark for summarizers that today's
   opaque-compaction systems cannot produce.
3. **Replay determinism.** Re-run the LLM against a historical commit's
   exact context with pinned tool results and measure how often behavior
   reproduces. Value: debugging and regression-testing agents.
4. **Counterfactual forks.** Branch at a decision point, vary one item,
   compare outcomes — evaluation harness and best-of-n execution in one
   mechanism.

### Baselines

JSONL append log (Claude Code's transcript format), SQLite event table,
LangGraph-style checkpointer. Same events, same operations. Compare storage,
reconstruction, retrieval, and the agent-quality arms above.

## Prior art

Event sourcing with content addressing is old. The near neighbors are
[Irmin](https://irmin.org/) (Git-model branchable store), Dolt (versioned
SQL), and Fossil (everything-in-one-DAG SCM). Agent-side, LangGraph
checkpointers and OpenHands' event stream snapshot per-step state, but into
opaque stores. The delta here: using the _DAG semantics_ — diffable
compaction, fork/merge provenance, content-addressed dedup and retrieval —
rather than Git as a file backup.

## Next steps

- batch commits per assistant turn (~3–4x fewer states) — the identified
  path to < 1.0x of the minimal event log
- pygit2 backend (sub-ms appends)
- tuned `repack` policy as part of the store, since default gc leaves 2–3x
  on the table for this workload
- `memory_search` as a real agent tool, wired into a live agent loop
  (e.g. Claude Agent SDK) recording actual sessions
- embedding sidecar keyed by blob SHA (grep-based `grep_history` does not
  scale to 72k blobs, so an index is mandatory at this size)
- agent-quality experiments (recoverable vs. lossy compaction) — the replay
  store now provides real contexts to run them against
