"""Local embedding model + deterministic chunker for the semantic sidecar.

Embeddings are derived state keyed by (blob SHA, chunk_no): content
addressing makes them compute-once-forever — there is no invalidation path,
only inserts. The chunker is deterministic, so chunk identities inherit that
guarantee.

The model runs locally (ONNX via fastembed, no torch): the archive contains
every secret that ever passed through a tool result, so nothing is sent to
an embedding API. First use downloads the model (~100 MB) to the HF cache.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, 512-token input window

# bge's input window is 512 tokens; ~1600 chars keeps chunks inside it so
# fastembed's silent truncation never drops content.
CHUNK_CHARS = 1600


def chunks(text: str) -> list[tuple[int, str]]:
    """Deterministic (offset, piece) split, preferring line boundaries."""
    if len(text) <= CHUNK_CHARS:
        return [(0, text)]
    out = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + CHUNK_CHARS // 2, end)
            if nl != -1:
                end = nl + 1
        out.append((start, text[start:end]))
        start = end
    return out


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class Embedder:
    """fastembed wrapper; import cost is paid lazily so ingest still works
    (FTS-only) on machines without the model."""

    name = MODEL_NAME

    def __init__(self):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(self.name)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(
            list(self._model.embed(texts, batch_size=256)), dtype=np.float32
        )
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed applies bge's retrieval instruction prefix
        vec = np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return _normalize(vec)


def load_embedder() -> Embedder | None:
    """None when fastembed isn't installed/usable — callers degrade to FTS."""
    try:
        return Embedder()
    except Exception:
        return None
