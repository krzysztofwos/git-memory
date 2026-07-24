"""Parse Claude Code JSONL transcripts into gitmem context events.

A transcript line of type user/assistant carries content blocks; each block
that enters the model's context becomes one (kind, role, content) event.
Extraction is deterministic, so a session ingested up to item N can be
resumed by appending events[N:] of a re-parse — the basis of incremental
ingestion of live sessions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".claude/projects"

_KIND_RE = re.compile(r"[^a-z_]")


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
    """Transcript file -> the context items it contributes, in order."""
    events: list[tuple[str, str, str]] = []

    def add(kind, role, content):
        if content:
            events.append((_KIND_RE.sub("_", kind.lower()) or "other", role, content))

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
            add("message", "user" if t == "user" else "assistant", content)
            continue
        for b in content or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                add(
                    "message", "user" if t == "user" else "assistant", b.get("text", "")
                )
            elif bt == "thinking":
                add("thinking", "assistant", b.get("thinking", ""))
            elif bt == "tool_use":
                add(
                    "tool_call",
                    "assistant",
                    json.dumps(
                        {"name": b.get("name"), "input": b.get("input")},
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                )
            elif bt == "tool_result":
                add("tool_result", "tool", flatten(b.get("content")))
            else:
                add(
                    bt or "other",
                    "user" if t == "user" else "assistant",
                    json.dumps(b, sort_keys=True, ensure_ascii=False),
                )
    return events


def slug(path: Path, root: Path = DEFAULT_ROOT) -> str:
    """Stable branch name for a transcript file."""
    rel = path.relative_to(root).with_suffix("")
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(rel)).strip("-")
    return s[-100:].strip("-")  # keep the unique uuid tail


def discover(root: Path = DEFAULT_ROOT) -> list[Path]:
    """All transcript files under a Claude Code projects dir, largest first."""
    return sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
