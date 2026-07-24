"""Install gitmem into Claude Code: a SessionStart ingest hook + a skill.

The hook keeps the archive fresh (incremental ingest is sub-second once the
initial ingest has run). The skill teaches Claude when and how to search it.
Both reference this environment's gitmem executable by absolute path, so no
PATH setup is needed. Idempotent: safe to re-run after moving the project.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HOOK_TIMEOUT = 10

SKILL_TEMPLATE = """\
---
name: gitmem
description: >-
  Search the verbatim archive of ALL past Claude Code sessions across every
  project. Use when the user references past work not in the current context:
  "have we/I done X before", "what was that error/fix/command", "when did we
  decide Y", "what did you do last week/in project Z", or when resuming work
  whose details are missing. Search before answering that something wasn't
  done or isn't known.
---

# gitmem: archive of past sessions

Every past Claude Code session (including subagents) is stored verbatim and
full-text indexed. Search returns provenance (session, position). Content is
retrieved byte-for-byte, so quote it rather than reconstructing from memory.

## Commands

```bash
{bin} search "connection pool exhausted"      # hybrid (keyword + semantic), all sessions
{bin} search --exact "MAX_RETRY_BACKOFF"      # FTS only: exact identifiers, fastest
{bin} search --semantic "weird build issue"   # vectors only: vague/paraphrased memories
{bin} search --kind tool_result "traceback"   # message|tool_call|tool_result|thinking
{bin} search --session theseus "deploy"       # filter by session-name substring
{bin} show <blob-sha>                         # verbatim original content
{bin} timeline <session> <seq> -n 5           # what surrounded a hit
{bin} sessions -n 20                          # recent sessions
{bin} ingest                                  # refresh archive (incremental)
```

FTS5 query syntax works: `"exact phrase"`, `term1 AND term2`, `deploy*`.
Hit markers: `[fts]` keyword match, `[sem]` semantic match, `[both]` — both
legs agree, highest confidence. If a hybrid search misses, retry --semantic
with a paraphrase, or --exact with a distinctive identifier.

## Discipline

- Search BEFORE claiming past work is unknown or lost to compaction.
- A hit is (session, seq, blob). Quote retrieved content verbatim and cite
  the session. Use `timeline` to understand context around a hit before
  drawing conclusions.
- The archive contains everything that passed through past sessions,
  including secrets in old tool output. Never paste retrieved secrets into
  responses, commits, or external services.
- The current session is indexed only up to the last ingest. Run
  `{bin} ingest` first if very recent context matters.
"""


def gitmem_bin() -> str:
    exe = Path(sys.argv[0]).resolve()
    if exe.name == "gitmem":
        return str(exe)
    found = shutil.which("gitmem")
    if found:
        return str(Path(found).resolve())
    return f"{sys.executable} -m gitmem.cli"


def install_skill(claude_dir: Path, bin_path: str) -> Path:
    skill = claude_dir / "skills/gitmem/SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(SKILL_TEMPLATE.format(bin=bin_path))
    return skill


def install_hook(claude_dir: Path, bin_path: str) -> tuple[Path, bool]:
    """Add a SessionStart ingest hook to settings.json. Returns (path, changed)."""
    settings_path = claude_dir / "settings.json"
    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    entries = settings.setdefault("hooks", {}).setdefault("SessionStart", [])
    # Backgrounded so session start never blocks on ingest/embedding. The
    # ingest lock makes overlapping runs no-op.
    command = f"nohup {bin_path} ingest --quiet >/dev/null 2>&1 &"
    for entry in entries:
        for hook in entry.get("hooks", []):
            if "gitmem" in hook.get("command", "") and " ingest" in hook.get(
                "command", ""
            ):
                if hook["command"] == command:
                    return settings_path, False
                hook["command"] = command  # project moved: update path
                _write_settings(settings_path, settings)
                return settings_path, True
    entries.append(
        {"hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT}]}
    )
    _write_settings(settings_path, settings)
    return settings_path, True


def _write_settings(path: Path, settings: dict) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(settings, indent=2) + "\n")


def cmd_setup(args) -> int:
    bin_path = gitmem_bin()
    skill = install_skill(args.claude_dir, bin_path)
    settings, changed = install_hook(args.claude_dir, bin_path)
    print(f"skill:    {skill}")
    print(
        f"hook:     {settings} ({'updated' if changed else 'already installed'}, "
        f"SessionStart -> gitmem ingest --quiet)"
    )
    print(f"binary:   {bin_path}")
    print(
        "note: run the initial `gitmem ingest --jobs 12` manually once. "
        "The hook only does cheap incremental updates."
    )
    return 0
