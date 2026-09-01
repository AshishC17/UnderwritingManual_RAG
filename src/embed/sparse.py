"""BM25 sparse vectors via FastEmbed.

Dense embeddings match on meaning and blur exact tokens together; BM25 matches
literal terms and understands no meaning at all. This document is full of exact
identifiers — rule codes 101-149, FC-nn criteria, "Table 7" — where a dense
vector's tolerance for near-misses is precisely the wrong behaviour.

FastEmbed emits term frequencies only. The IDF half of BM25 is computed by
Qdrant from collection statistics, which is why the collection declares
`Modifier.IDF` on its sparse vector — without it, scoring silently drops the
term-rarity signal that makes BM25 work.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import models

MODEL = "Qdrant/bm25"


@lru_cache(maxsize=1)
def _model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(MODEL)


def _to_sparse(vec) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(i) for i in vec.indices],
        values=[float(v) for v in vec.values],
    )


def embed_documents(texts: list[str]) -> list[models.SparseVector]:
    """Document-side vectors: BM25 term frequencies with saturation applied."""
    return [_to_sparse(v) for v in _model().embed(texts)]


def embed_query(text: str) -> models.SparseVector:
    """Query-side vectors: term presence, not frequency. Asymmetric with the
    document side on purpose — a query term appearing twice should not double
    its weight."""
    return _to_sparse(next(iter(_model().query_embed([text]))))
