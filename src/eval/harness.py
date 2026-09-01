"""Run retrieval configurations against an eval split.

Query embeddings are computed once up front and shared by every configuration —
the same vector serves dense, hybrid, and hybrid+rerank. Only embedding and
reranking touch the Voyage API; Qdrant search and BM25 are local and free, so
the three non-rerank configurations cost nothing to add.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.embed.embedder import embed_query
from src.embed.sparse import embed_query as embed_query_sparse
from src.rerank.reranker import rerank
from src.store import qdrant_store as qs

CONFIGS = ["dense", "sparse", "hybrid", "hybrid_rerank"]


@dataclass
class Retrieval:
    chunk_ids: list[str]
    latency_ms: float


def prepare_queries(cases: list[dict], spacing_s: int, verbose: bool = True) -> dict:
    """Embed every query once. Cached, so only the first run pays the rate limit."""
    from src.embed.embedder import _is_cached, CACHE_DIR, MODEL, DIMS

    vectors: dict[str, tuple] = {}
    for i, case in enumerate(cases):
        q = case["question"]
        cached = _is_cached([q], MODEL, DIMS, "query", CACHE_DIR)
        if i and not cached and spacing_s:
            if verbose:
                print(f"    rate-limit wait {spacing_s}s...", flush=True)
            time.sleep(spacing_s)
        vectors[case["id"]] = (embed_query(q), embed_query_sparse(q))
        if verbose:
            mark = "cached" if cached else "embedded"
            print(f"    [{i + 1}/{len(cases)}] {case['id']} {mark}", flush=True)
    return vectors


def retrieve(
    client,
    config: str,
    dense_vec,
    sparse_vec,
    k: int,
    candidates: int,
    query: str | None = None,
    payload_by_id: dict | None = None,
) -> Retrieval:
    t0 = time.perf_counter()

    if config == "dense":
        hits = qs.search(client, dense_vec, limit=k)
        ids = [h.payload["chunk_id"] for h in hits]
    elif config == "sparse":
        hits = qs.search_sparse(client, sparse_vec, limit=k)
        ids = [h.payload["chunk_id"] for h in hits]
    elif config == "hybrid":
        hits = qs.search_hybrid(client, dense_vec, sparse_vec, limit=k)
        ids = [h.payload["chunk_id"] for h in hits]
    elif config == "hybrid_rerank":
        hits = qs.search_hybrid(
            client, dense_vec, sparse_vec, limit=candidates,
            prefetch_limit=candidates * 2,
        )
        payloads = [h.payload for h in hits]
        ranked = rerank(query, payloads, top_k=k)
        ids = [p["chunk_id"] for p, _ in ranked]
    else:
        raise ValueError(f"unknown config: {config}")

    return Retrieval(chunk_ids=ids, latency_ms=(time.perf_counter() - t0) * 1000)
