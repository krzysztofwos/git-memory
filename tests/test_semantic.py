import hashlib

import numpy as np
import pytest

from gitmem.embed import CHUNK_CHARS, chunks
from gitmem.index import SearchIndex


class FakeEmbedder:
    """Deterministic vectors from content hashes; identical text -> identical
    vector, so self-similarity is 1.0. Exercises storage/ranking/fusion
    mechanics without downloading a model."""

    name = "fake-model"

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(32).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def emb():
    return FakeEmbedder()


def test_chunker_deterministic_and_bounded():
    text = "x" * 10 + "\n" + ("word " * 2000)
    cs = chunks(text)
    assert cs == chunks(text)
    assert all(len(piece) <= CHUNK_CHARS for _, piece in cs)
    assert "".join(p for _, p in cs) == text
    assert chunks("short") == [(0, "short")]


def test_vectors_stored_once_and_ranked(tmp_path, emb):
    index = SearchIndex(tmp_path / "i.sqlite")
    docs = {"sha-a": "the cat sat on the mat", "sha-b": "kubernetes pod crashloop"}
    rows = []
    for sha, text in docs.items():
        for no, (off, piece) in enumerate(chunks(text)):
            rows.append((sha, no, off, emb.embed_docs([piece])[0]))
    index.add_vectors(emb.name, rows)
    assert index.vector_count(emb.name) == 2
    assert index.missing_vectors({"sha-a", "sha-b", "sha-c"}, emb.name) == {"sha-c"}

    ranked = index.semantic_ranked(emb.embed_query("the cat sat on the mat"),
                                   emb.name, limit=2)
    # fp16 storage quantizes vectors, so self-similarity is ~1.0, not exactly
    assert ranked[0][0] == "sha-a" and ranked[0][2] == pytest.approx(1.0, abs=1e-2)


def test_hybrid_fuses_both_legs(tmp_path, emb):
    index = SearchIndex(tmp_path / "i.sqlite")
    docs = {
        "sha-kw": "grep finds this keyword needle exactly",
        "sha-sem": "completely different words about felines on rugs",
    }
    index.add("sess", [(1, "message", "user", "sha-kw"), (2, "message", "user", "sha-sem")],
              docs)
    index.add_vectors(emb.name, [
        (sha, 0, 0, emb.embed_docs([text])[0]) for sha, text in docs.items()
    ])

    # keyword-only query: FTS leg finds it; semantic leg ranks its own text top
    hits = index.hybrid_search(
        "needle", emb.embed_query("completely different words about felines on rugs"),
        emb.name, limit=2)
    assert {h.sha for h in hits} == {"sha-kw", "sha-sem"}
    origins = {h.sha: h.origin for h in hits}
    assert origins["sha-kw"] in ("fts", "both")
    by_sha = {h.sha: h for h in hits}
    assert by_sha["sha-sem"].snippet is None or "felines" in by_sha["sha-sem"].snippet

    # a sha found by BOTH legs outranks single-leg hits under RRF
    both = index.hybrid_search("needle", emb.embed_query(docs["sha-kw"]), emb.name, limit=2)
    assert both[0].sha == "sha-kw" and both[0].origin == "both"


def test_hybrid_without_vectors_degrades_to_fts(tmp_path):
    index = SearchIndex(tmp_path / "i.sqlite")
    index.add("sess", [(1, "message", "user", "sha-1")], {"sha-1": "plain text here"})
    hits = index.hybrid_search("plain", None, "fake-model", limit=5)
    assert len(hits) == 1 and hits[0].origin == "fts"
