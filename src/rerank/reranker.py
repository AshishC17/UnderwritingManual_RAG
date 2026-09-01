"""Cross-encoder reranking via Voyage.

The retrievers before this stage are *bi-encoders*: query and document are
encoded separately and compared as vectors, so the model never sees the two
together. That is what makes them fast enough to search a whole collection, and
also what caps their accuracy — the document's vector was computed with no idea
what would be asked of it.

A cross-encoder encodes query and document *jointly*, so attention runs across
both. Far better relevance judgements, but it costs one model pass per candidate
and cannot be run over a collection. Hence two stages: retrieve broadly and
cheaply, then rerank a shortlist expensively.

Cached on (model, query, candidate texts) — the output depends on the exact
candidate set, not the query alone, so a query-only key would be wrong the
moment retrieval returns something different.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL = "rerank-2.5"
CACHE_DIR = "data/interim/reranks"


class MissingCredentials(RuntimeError):
    pass


def _key(model: str, query: str, texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{query}|".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:20]


def _client():
    import voyageai

    if not os.environ.get("VOYAGE_API_KEY"):
        raise MissingCredentials("VOYAGE_API_KEY is not set")
    return voyageai.Client()


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    model: str = MODEL,
    cache_dir: str = CACHE_DIR,
) -> list[tuple[dict, float]]:
    """Reorder `candidates` (chunk payload dicts) by joint relevance to `query`.

    Returns (chunk, relevance_score) pairs, most relevant first. Scores are
    absolute 0-1 relevance judgements, not similarities — unlike the fused RRF
    ranks coming in, they are comparable across queries and can carry a threshold.
    """
    texts = [c["text"] for c in candidates]
    cache = Path(cache_dir) / f"{_key(model, query, texts)}.json"

    if cache.exists():
        scored = json.loads(cache.read_text())
    else:
        result = _client().rerank(query=query, documents=texts, model=model)
        scored = [[r.index, r.relevance_score] for r in result.results]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scored))

    out = [(candidates[i], score) for i, score in scored]
    return out[:top_k] if top_k else out
