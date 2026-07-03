from gitmem import MemoryStore


def test_append_materialize_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "s.git")
    m = store.session("main")
    m.append("message", "system", "sys prompt")
    m.append("message", "user", "hello")
    m.append("tool_result", "tool", "output " * 50)
    items = m.materialize()
    assert [(i.seq, i.kind, i.role) for i in items] == [
        (1, "message", "system"), (2, "message", "user"), (3, "tool_result", "tool"),
    ]
    assert items[1].content == "hello"


def test_dedup_identical_content(tmp_path):
    store = MemoryStore(tmp_path / "s.git")
    m = store.session("main")
    m.append("tool_result", "tool", "same bytes")
    m.append("tool_result", "tool", "same bytes")
    a, b = m.materialize()
    assert a.blob == b.blob and a.seq != b.seq


def test_compact_is_nondestructive(tmp_path):
    store = MemoryStore(tmp_path / "s.git")
    m = store.session("main")
    for i in range(5):
        m.append("message", "user", f"needle-{i}")
    pre = m.tip
    m.compact(2, 5, "summary of 2..5")
    live = m.materialize()
    assert [(i.seq, i.kind) for i in live] == [(1, "message"), (2, "summary")]
    old = m.materialize(pre)
    assert len(old) == 5 and old[3].content == "needle-3"
    hits = store.grep_history("needle-3")
    assert hits and store.retrieve(hits[0].blob) == "needle-3"


def test_compact_across_bucket_boundary(tmp_path):
    store = MemoryStore(tmp_path / "s.git")
    m = store.session("main")
    for i in range(300):
        m.append("message", "user", f"m{i}")
    m.compact(200, 300, "boundary summary")
    assert [i.seq for i in m.materialize()] == list(range(1, 200)) + [200]


def test_fork_absorb_merge_parents(tmp_path):
    store = MemoryStore(tmp_path / "s.git")
    m = store.session("main")
    m.append("message", "user", "task")
    sub = m.fork("sub")
    sub.append("message", "user", "subtask detail")
    m.absorb(sub, "sub verdict")
    parents = store.git("log", "-1", "--format=%P", m.tip).split()
    assert parents == [store.git("rev-parse", "main~1").strip(), sub.tip]
    assert m.materialize()[-1].content == "sub verdict"


def test_cold_reopen_resumes_seq_and_tokens(tmp_path):
    store = MemoryStore(tmp_path / "s.git")
    m = store.session("main")
    m.append("message", "user", "one")
    m.append("message", "user", "two")
    tokens = m.token_total()
    m2 = MemoryStore(tmp_path / "s.git").session("main")
    assert m2.next_seq == 3 and m2.token_total() == tokens
    m2.append("message", "user", "three")
    assert [i.seq for i in m2.materialize()] == [1, 2, 3]
