import json

import pytest

from gitmem import MemoryStore
from gitmem.index import SearchIndex
from gitmem.ingest import ingest_all, ingest_file
from gitmem.transcript import extract_events, slug


def transcript_line(t, blocks):
    return json.dumps({"type": t, "message": {"role": t, "content": blocks}})


def write_transcript(path, n_turns, tag=""):
    lines = []
    for i in range(n_turns):
        lines.append(transcript_line("user", [{"type": "text", "text": f"ask{tag} {i}"}]))
        lines.append(transcript_line("assistant", [
            {"type": "text", "text": f"answer{tag} {i}"},
            {"type": "tool_use", "name": "Bash", "input": {"command": f"run{tag} {i}"}},
        ]))
        lines.append(transcript_line("user", [
            {"type": "tool_result", "content": f"result{tag} {i} unique-marker-{tag}{i}"},
        ]))
        lines.append(json.dumps({"type": "mode", "mode": "noise"}))  # non-context line
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "projects"
    store = MemoryStore(tmp_path / "store.git")
    index = SearchIndex(tmp_path / "index.sqlite")
    return root, store, index


def test_extract_skips_non_context_lines(tmp_path):
    f = tmp_path / "projects/p/a.jsonl"
    write_transcript(f, 2)
    events = extract_events(f)
    assert len(events) == 8  # 4 context blocks per turn
    assert {k for k, _, _ in events} == {"message", "tool_call", "tool_result"}


def test_ingest_is_incremental_at_item_level(env):
    root, store, index = env
    f = root / "proj/session-1.jsonl"
    write_transcript(f, 3)
    session, appended = ingest_file(store, f, root)
    assert appended == 12
    write_transcript(f, 5)  # session continues: same prefix, new tail
    session, appended = ingest_file(store, f, root)
    assert appended == 8
    items = store.session(slug(f, root)).materialize()
    assert len(items) == 20
    assert items[0].content == "ask 0" and items[-1].content.startswith("result 4")


def test_ingest_all_watermark_and_search(env):
    root, store, index = env
    write_transcript(root / "proj/a.jsonl", 2, tag="A")
    write_transcript(root / "proj/b.jsonl", 2, tag="B")
    stats = ingest_all(store, index, root=root)
    assert stats.items_appended == 16 and stats.files_ingested == 2

    hits = index.search("unique-marker-A1")
    assert len(hits) == 1
    (session, seq, kind), = hits[0].occurrences
    assert "proj-a" in session and kind == "tool_result"
    assert "unique-marker-A1" in store.retrieve(hits[0].sha)

    # unchanged files are skipped entirely by the mtime watermark
    stats2 = ingest_all(store, index, root=root)
    assert stats2.files_seen == 0 and stats2.items_appended == 0


def test_search_filters_and_fallback(env):
    root, store, index = env
    write_transcript(root / "proj/a.jsonl", 2)
    f = root / "proj/b.jsonl"
    f.write_text(transcript_line(
        "user", [{"type": "text", "text": "failed calling weird(query here"}]) + "\n")
    ingest_all(store, index, root=root)
    assert index.search("ask", kind="tool_result") == []
    assert index.search("ask", kind="message")
    hits = index.search("weird(query")  # invalid FTS5 syntax -> literal fallback
    assert hits and "weird(query" in store.retrieve(hits[0].sha)


def test_concurrent_ingest_noops_via_lock(tmp_path):
    import fcntl

    from gitmem import cli

    home = tmp_path / "gm"
    projects = tmp_path / "projects"
    projects.mkdir()
    home.mkdir()
    holder = open(home / ".ingest.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rc = cli.main(["--home", str(home), "ingest", "--quiet",
                   "--no-embed", "--projects", str(projects)])
    assert rc == 0
    assert not (home / "store.git").exists()  # lock stopped it before any work
