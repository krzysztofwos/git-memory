"""gitmem: Git as the storage and operation log for LLM agent memory.

Core invariant: the agent's context window is a pure function of a commit SHA.
The tree at HEAD *is* the context. Every mutation (append a message, record a
tool call/result, compact, absorb a subagent) is a new commit. History is
never rewritten, so compaction is non-destructive: pre-compaction states stay
reachable as ancestors, and `git diff` shows exactly what a summary replaced.

Object layout of each commit's tree (fanout: buckets of 256 items, so an
append rewrites one bucket subtree + the root instead of an O(N) flat tree --
a flat layout has quadratic total tree bytes, measured 3.4x a minimal event
log on real 10k-item transcripts):

    items/0000/000001.message.system.md   # blob = raw content, nothing else
    items/0000/000002.message.user.md
    items/0000/000002.summary.assistant.md  # a compaction reused seq 2's slot
    items/0003/000841.tool_result.tool.md
    state.json                              # {"next_seq": N}

Per-item metadata (seq, kind, role) lives in the *filename* and operation
metadata lives in *commit message trailers*, so a blob is exactly the raw
content. That makes Git's content addressing do deduplication for free:
two identical tool results (e.g. re-reading an unchanged file) are one blob
no matter how many context states reference them.

Implementation uses git plumbing via subprocess (hash-object, mktree,
commit-tree, update-ref) against a bare repo -- zero dependencies. A real
system would use libgit2/pygit2 or gitoxide for sub-millisecond appends.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path

ITEM_RE = re.compile(r"^items/(?:\d+/)?(\d{6})\.([a-z_]+)\.([a-z_]+)\.md$")
BUCKET_SIZE = 256


def _bucket(seq: int) -> str:
    return f"{seq // BUCKET_SIZE:04d}"


def _item_path(seq: int, kind: str, role: str) -> str:
    return f"items/{_bucket(seq)}/{seq:06d}.{kind}.{role}.md"


def est_tokens(text: str) -> int:
    """Crude token estimate (len/4). Pluggable, only used for bookkeeping."""
    return max(1, len(text) // 4)


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Item:
    seq: int
    kind: str  # message | tool_call | tool_result | summary
    role: str  # system | user | assistant | tool
    content: str
    blob: str  # blob SHA (stable content address)

    @property
    def tokens(self) -> int:
        return est_tokens(self.content)

    def header(self) -> str:
        return f"[#{self.seq} {self.kind}/{self.role}]"


@dataclass(frozen=True)
class Hit:
    """A grep hit anywhere in history, including compacted-away content."""

    path: str
    blob: str
    snippet: str


class MemoryStore:
    """A bare Git repository holding one or more agent sessions (branches)."""

    def __init__(self, path: str | Path, fresh: bool = False):
        self.path = Path(path)
        if fresh and self.path.exists():
            shutil.rmtree(self.path)
        if not (self.path / "HEAD").exists():
            self.path.mkdir(parents=True, exist_ok=True)
            subprocess.run(  # nosec B603 B607
                ["git", "init", "--bare", "-q", str(self.path)], check=True
            )
            self.git("config", "user.name", "gitmem")
            self.git("config", "user.email", "gitmem@localhost")

    def git(self, *args: str, data: str | None = None) -> str:
        p = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(self.path), *args],
            check=False,
            input=data,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {p.stderr.strip()}")
        return p.stdout

    def git_bytes(self, *args: str, data: bytes | None = None) -> bytes:
        p = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(self.path), *args],
            check=False,
            input=data,
            capture_output=True,
        )
        if p.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {p.stderr.decode()}")
        return p.stdout

    def session(self, name: str) -> Session:
        return Session(self, name)

    # ---- retrieval over *everything ever stored*, compacted or not ----

    def grep_history(self, pattern: str, context: int = 0) -> list[Hit]:
        """Search every item blob reachable from any ref -- including content
        that was compacted out of every live context. Nothing is ever lost."""
        rx = re.compile(pattern)
        seen: dict[str, str] = {}  # blob sha -> first path seen
        for line in self.git("rev-list", "--all", "--objects").splitlines():
            sha, _, path = line.partition(" ")
            if ITEM_RE.match(path) and sha not in seen:
                seen[sha] = path
        hits: list[Hit] = []
        for sha, path in seen.items():
            content = self.git("cat-file", "blob", sha)
            lines = content.splitlines()
            for i, ln in enumerate(lines):
                if rx.search(ln):
                    lo, hi = max(0, i - context), i + context + 1
                    hits.append(Hit(path, sha, "\n".join(lines[lo:hi])))
                    break
        return hits

    def retrieve(self, blob: str) -> str:
        """Fetch full original content by content address."""
        return self.git("cat-file", "blob", blob)

    def cat_batch(self, shas: list[str]) -> dict[str, str]:
        """Read many blobs in one git process."""
        if not shas:
            return {}
        out = self.git_bytes(
            "cat-file", "--batch", data="".join(s + "\n" for s in shas).encode()
        )
        result: dict[str, str] = {}
        pos = 0
        for sha in shas:
            nl = out.index(b"\n", pos)
            header = out[pos:nl].decode()
            size = int(header.split()[2])
            result[sha] = out[nl + 1 : nl + 1 + size].decode()
            pos = nl + 1 + size + 1  # trailing newline
        return result

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())

    def gc(self) -> None:
        self.git("gc", "--quiet", "--aggressive")


class Session:
    """One agent session = one branch. Each event/operation = one commit."""

    def __init__(self, store: MemoryStore, name: str):
        self.store = store
        self.name = name
        self.ref = f"refs/heads/{name}"
        self.tip: str | None = None
        self.files: dict[str, str] = {}  # items/<bucket>/<fname> -> blob sha
        self._tok: dict[str, int] = {}  # same keys -> token estimate
        self._bucket_cache: dict[str, str] = {}  # bucket name -> clean tree sha
        self._dirty: set[str] = set()  # buckets to rebuild at next commit
        self.next_seq = 1
        try:
            self.tip = store.git("rev-parse", "--verify", "-q", self.ref).strip()
        except GitError:
            return
        for line in store.git("ls-tree", "-r", "-t", self.tip).splitlines():
            meta, _, path = line.partition("\t")
            _mode, otype, sha = meta.split()
            if otype == "tree" and re.fullmatch(r"items/\d+", path):
                self._bucket_cache[path.split("/")[1]] = sha
            elif otype == "blob" and (m := ITEM_RE.match(path)):
                if path.count("/") == 1:  # legacy flat layout: migrate in-memory
                    path = _item_path(int(m.group(1)), m.group(2), m.group(3))
                    self._dirty.add(_bucket(int(m.group(1))))
                self.files[path] = sha
            elif path == "state.json":
                self.next_seq = json.loads(store.git("cat-file", "blob", sha))[
                    "next_seq"
                ]
        blobs = self.store.cat_batch(sorted(set(self.files.values())))
        self._tok = {p: est_tokens(blobs[sha]) for p, sha in self.files.items()}

    # ---- write path ----

    def _write_blob(self, content: str) -> str:
        return self.store.git("hash-object", "-w", "--stdin", data=content).strip()

    def _commit(self, message: str, extra_parents: list[str] | None = None) -> str:
        for b in self._dirty:
            prefix = f"items/{b}/"
            entries = "".join(
                f"100644 blob {sha}\t{path.removeprefix(prefix)}\n"
                for path, sha in sorted(self.files.items())
                if path.startswith(prefix)
            )
            if entries:
                self._bucket_cache[b] = self.store.git("mktree", data=entries).strip()
            else:
                self._bucket_cache.pop(b, None)
        self._dirty.clear()
        items_tree = self.store.git(
            "mktree",
            data="".join(
                f"040000 tree {sha}\t{b}\n"
                for b, sha in sorted(self._bucket_cache.items())
            ),
        ).strip()
        state_blob = self._write_blob(json.dumps({"next_seq": self.next_seq}) + "\n")
        root = self.store.git(
            "mktree",
            data=(
                f"040000 tree {items_tree}\titems\n"
                f"100644 blob {state_blob}\tstate.json\n"
            ),
        ).strip()
        parents: list[str] = []
        if self.tip:
            parents += ["-p", self.tip]
        for p in extra_parents or []:
            parents += ["-p", p]
        new = self.store.git("commit-tree", root, *parents, "-m", message).strip()
        if self.tip:
            self.store.git("update-ref", self.ref, new, self.tip)  # CAS
        else:
            self.store.git("update-ref", self.ref, new)
        self.tip = new
        return new

    def append(self, kind: str, role: str, content: str) -> str:
        seq = self.next_seq
        self.next_seq += 1
        fname = _item_path(seq, kind, role)
        self.files[fname] = self._write_blob(content)
        self._tok[fname] = est_tokens(content)
        self._dirty.add(_bucket(seq))
        total = self.token_total()
        msg = (
            f"append {kind} #{seq:06d} (+{est_tokens(content)} tok, total {total})\n\n"
            f"Op: append\nKind: {kind}\nSeq: {seq}\nTokens-Total: {total}\n"
        )
        return self._commit(msg)

    def compact(self, start_seq: int, end_seq: int, summary: str) -> str:
        """Replace items in [start_seq, end_seq] with one summary item.

        Non-destructive: the removed blobs remain reachable via the parent
        commit. `git diff HEAD~1 HEAD` shows exactly what the summary elides.
        """
        before = self.token_total()
        victims = [
            p
            for p in self.files
            if (m := ITEM_RE.match(p)) and start_seq <= int(m.group(1)) <= end_seq
        ]
        if not victims:
            raise ValueError(f"no items in range {start_seq}..{end_seq}")
        for p in victims:
            del self.files[p]
            del self._tok[p]
            self._dirty.add(p.split("/")[1])
        fname = _item_path(start_seq, "summary", "assistant")
        self.files[fname] = self._write_blob(summary)
        self._tok[fname] = est_tokens(summary)
        self._dirty.add(_bucket(start_seq))
        after = self.token_total()
        msg = (
            f"compact #{start_seq:06d}..#{end_seq:06d}: {len(victims)} items -> 1 "
            f"summary ({before} -> {after} tok)\n\n"
            f"Op: compact\nReplaced: {start_seq}..{end_seq}\n"
            f"Items-Removed: {len(victims)}\nTokens-Total: {after}\n"
        )
        return self._commit(msg)

    def fork(self, name: str) -> Session:
        """Branch a child context (e.g. a subagent) from the current state."""
        if not self.tip:
            raise GitError("cannot fork an empty session")
        self.store.git("branch", name, self.tip)
        return Session(self.store, name)

    def absorb(self, child: Session, summary: str) -> str:
        """Merge a subagent back: append its summary, with a merge commit
        whose second parent is the child's tip -- full provenance of where
        the child forked and everything it did."""
        seq = self.next_seq
        self.next_seq += 1
        fname = _item_path(seq, "summary", "assistant")
        self.files[fname] = self._write_blob(summary)
        self._tok[fname] = est_tokens(summary)
        self._dirty.add(_bucket(seq))
        total = self.token_total()
        msg = (
            f"absorb subagent '{child.name}' as #{seq:06d} (total {total} tok)\n\n"
            f"Op: absorb\nChild: {child.name}\nSeq: {seq}\nTokens-Total: {total}\n"
        )
        return self._commit(msg, extra_parents=[child.tip] if child.tip else None)

    # ---- read path ----

    def materialize(self, rev: str | None = None) -> list[Item]:
        """Reconstruct the exact context at any commit (default: tip)."""
        rev = rev or self.tip
        if rev is None:
            return []
        rev = self.store.git("rev-parse", rev).strip()
        entries = []  # (seq, kind, role, sha, path)
        for line in self.store.git("ls-tree", "-r", rev).splitlines():
            meta, _, path = line.partition("\t")
            m = ITEM_RE.match(path)
            if m:
                entries.append(
                    (int(m.group(1)), m.group(2), m.group(3), meta.split()[2])
                )
        entries.sort()
        blobs = self.store.cat_batch(sorted({e[3] for e in entries}))
        return [
            Item(seq, kind, role, blobs[sha], sha) for seq, kind, role, sha in entries
        ]

    def prompt_text(self, rev: str | None = None) -> str:
        """The string an LLM call would be built from -- a pure function of rev."""
        return "\n\n".join(
            f"{it.header()}\n{it.content}" for it in self.materialize(rev)
        )

    def token_total(self) -> int:
        # Per file, not per unique blob: a deduplicated blob still occupies
        # context tokens at every position it appears.
        return sum(self._tok.values())

    def log_oneline(self, *extra: str) -> str:
        return self.store.git("log", "--oneline", *extra, self.ref)
