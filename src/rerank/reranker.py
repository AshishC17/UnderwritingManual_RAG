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
import time
from pathlib import Path

MODEL = "rerank-2.5"
CACHE_DIR = "data/interim/reranks"

# The free tier allows 3 requests and 10K tokens per minute, and reranking a
# shortlist spends a large share of that in one call. The SDK's own retry gives
# up well inside a 60s window, so back off past the whole window instead.
RETRY_WAITS_S = [65, 65, 125]

# A 25-candidate shortlist of ~200-token chunks measures ~8.2K tokens — 82% of a
# free-tier minute in one call. Retrying an over-budget request cannot succeed
# and burns the budget it is waiting for, so trim the shortlist to fit instead.
# Raise this once rate limits allow; it caps recall, since the reranker can only
# reorder what it is given.
MAX_RERANK_TOKENS = 7500


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


def _rerank_with_backoff(query: str, texts: list[str], model: str, verbose: bool = True):
    """Call rerank, waiting out the rate-limit window rather than failing."""
    import voyageai

    client = _client()
    for attempt, wait in enumerate([0, *RETRY_WAITS_S]):
        if wait:
            if verbose:
                print(f"      rate limited; waiting {wait}s "
                      f"(attempt {attempt}/{len(RETRY_WAITS_S)})", flush=True)
            time.sleep(wait)
        try:
            return client.rerank(query=query, documents=texts, model=model)
        except voyageai.error.RateLimitError:
            if attempt == len(RETRY_WAITS_S):
                raise
    raise RuntimeError("unreachable")


def _fit_budget(candidates: list[dict], max_tokens: int) -> list[dict]:
    """Keep the highest-ranked candidates that fit the token budget.

    Trimming from the tail is safe: candidates arrive in retrieval-rank order, so
    the ones dropped are those retrieval already judged least promising.
    """
    kept, used = [], 0
    for c in candidates:
        cost = c.get("token_count") or max(1, len(c["text"]) // 4)
        if kept and used + cost > max_tokens:
            break
        kept.append(c)
        used += cost
    return kept


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    model: str = MODEL,
    cache_dir: str = CACHE_DIR,
    max_tokens: int = MAX_RERANK_TOKENS,
) -> list[tuple[dict, float]]:
    """Reorder `candidates` (chunk payload dicts) by joint relevance to `query`.

    Returns (chunk, relevance_score) pairs, most relevant first. Scores are
    absolute 0-1 relevance judgements, not similarities — unlike the fused RRF
    ranks coming in, they are comparable across queries and can carry a threshold.
    """
    candidates = _fit_budget(candidates, max_tokens)
    texts = [c["text"] for c in candidates]
    cache = Path(cache_dir) / f"{_key(model, query, texts)}.json"

    if cache.exists():
        scored = json.loads(cache.read_text())
    else:
        result = _rerank_with_backoff(query, texts, model)
        scored = [[r.index, r.relevance_score] for r in result.results]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scored))

    out = [(candidates[i], score) for i, score in scored]
    return out[:top_k] if top_k else out
