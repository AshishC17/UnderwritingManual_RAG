"""Paired, source-excerpt retrieval evaluation of two immutable index builds.

Same questions, scopes, k, context token budget and reranker on both sides.
No LLM generation/judging, decomposition or semantic cache. Quality metrics are
NOT orchestration or groundedness scores. Cached-only unless --allow-provider.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from src.embed import embedder, sparse
from src.rerank import reranker
from src.eval.excerpts import map_evidence
from src.ingest.chunker import n_tokens
from src.store import index_builds as ib, qdrant_store as qs
from src.store.index_context import bind
from scripts.run_chunking_eval import score_case


def select_context(chunks, k, budget):
    kept, used = [], 0
    # Preserve ranking, do not split a row to squeeze it into the budget.
    for chunk in chunks[:k]:
        cost = n_tokens(chunk["text"])
        if used + cost > budget:
            break
        kept.append(chunk)
        used += cost
    return kept, used


def evaluate(client, active, candidate, excerpts_path, cases_path, *, k=10, candidates=24,
             token_budget=4000, allow_provider=False, use_reranker=True, sparse_only=False, root=ib.BUILDS):
    if not 1 <= k <= candidates or token_budget <= 0:
        raise ValueError("require 1 <= k <= candidates and a positive context token budget")
    builds = [ib.read_build(b, root) for b in (active, candidate)]
    refs = json.loads(Path(excerpts_path).read_text())
    if refs.get("schema_version") != "2.0":
        raise ValueError("use source-anchored excerpt schema v2")
    raw_cases = json.loads(Path(cases_path).read_text())
    cases = raw_cases["cases"] if isinstance(raw_cases, dict) else raw_cases
    by_case = {}
    for g in refs["groups"]:
        by_case.setdefault(g["case"], []).append(g)
    selected = [c for c in cases if c["id"] in by_case]
    if not selected or {c["id"] for c in selected} != set(by_case):
        raise ValueError("evaluation must contain every case referenced by the anchors")
    report = {"schema_version": 1, "created_at": ib.now(), "active_build": active, "candidate_build": candidate,
              "active_manifest_sha256": builds[0]["manifest_sha256"],
              "candidate_manifest_sha256": builds[1]["manifest_sha256"],
              "excerpts_sha256": ib.digest(Path(excerpts_path).read_bytes()), "reference_status": refs["status"],
              "cases_sha256": ib.digest(Path(cases_path).read_bytes()),
              "evaluation_code_sha256": ib.digest(Path(__file__).read_bytes()),
              "config": {"k": k, "candidates": candidates, "context_token_budget": token_budget,
                         "retrieval": ("bm25" if sparse_only else "hybrid") + ("_rerank" if use_reranker else ""), "semantic_cache": False,
                         "reranker": reranker.MODEL if use_reranker else None,
                         "tokenizer": "cl100k_base", "allow_provider": allow_provider},
              "not_measured": ["generation completeness", "groundedness", "orchestration", "production latency", "billed cost"],
              "builds": {}}
    for m in builds:
        build_id = m["build_id"]
        ib.validate_collection(client, build_id, root)
        sources = {s["source_doc"]: s["sha256"] for s in m["sources"] if s["kind"] == "path"}
        for item in refs["excerpts"].values():
            if sources.get(item["document"]) != refs["source_sha256"].get(item["document"]):
                raise ValueError("source PDF changed: re-author affected ground truth; do not re-label silently")
        chunks = json.loads((ib.build_dir(build_id, root) / "chunks.json").read_text())
        mapping = map_evidence(refs, chunks)
        rows = {}
        binding = {**ib.binding_for(build_id, root), "disable_semantic_cache": True}
        with bind(binding):
            for case in selected:
                query = case["question"]
                model, dims = m["dense"]["model"], m["dense"]["dims"]
                cached = embedder._is_cached([query], model, dims, "query", embedder.CACHE_DIR)
                if not sparse_only and not allow_provider and not cached:
                    raise ValueError(f"{case['id']}: query embedding not cached; opt in to provider calls")
                started = time.perf_counter()
                sv = sparse.embed_query(query)
                scope = case["expected_scope"]
                filt = qs.scope_filter(lender_id=scope["lender_id"], document_version=scope["document_version"],
                                       document_id=scope.get("document_id"))
                if sparse_only:
                    hits = qs.search_sparse(client, sv, limit=candidates, query_filter=filt)
                else:
                    dv = embedder.embed_query(query, model=model, dims=dims)
                    hits = qs.search_hybrid(client, dv, sv, limit=candidates, prefetch_limit=2*candidates, query_filter=filt)
                ranked = [p.payload for p in hits]
                rerank_cached = None
                if use_reranker and ranked:
                    fitted = reranker._fit_budget(ranked, reranker.MAX_RERANK_TOKENS)
                    cache = Path(reranker.CACHE_DIR) / (reranker._key(reranker.MODEL, query, [c["text"] for c in fitted]) + ".json")
                    rerank_cached = cache.exists()
                    if not allow_provider and not rerank_cached:
                        raise ValueError(f"{case['id']}: rerank not cached; opt in to provider calls or use --no-rerank")
                    ranked = [chunk for chunk, score in reranker.rerank(query, ranked)]
                kept, used = select_context(ranked, k, token_budget)
                result = score_case(case, refs["excerpts"], by_case[case["id"]], kept)
                result["greedy_support_chunk_count"] = result.pop("chunks_needed")
                result.update(retrieved_chunk_ids=[c["chunk_id"] for c in kept], context_tokens=used,
                              elapsed_ms=round((time.perf_counter()-started)*1000, 2),
                              query_embedding_cached=None if sparse_only else cached, rerank_cached=rerank_cached)
                # Unmatched does NOT mean irrelevant: the answer-key excerpts
                # are not exhaustive relevance labels for every returned chunk.
                matched = {cid for r in result["excerpts"].values() for cid in r.get("all_matching_chunk_ids", r["chunk_ids"])}
                result["anchor_matched_chunk_fraction"] = len(matched)/len(kept) if kept else 0.
                rows[case["id"]] = result
                print(f"{build_id} {case['id']}: groups={result['group_coverage']:.3f} full={result['full_coverage']} tokens={used}", flush=True)
        report["builds"][build_id] = {"mapping": mapping, "cases": rows,
            "mean_group_coverage": sum(r["group_coverage"] for r in rows.values())/len(rows),
            "full_coverage_rate": sum(r["full_coverage"] for r in rows.values())/len(rows)}
    a, b = report["builds"][active], report["builds"][candidate]
    report["regressions"] = [cid for cid in a["cases"] if b["cases"][cid]["group_coverage"] < a["cases"][cid]["group_coverage"]]
    report["delta_mean_group_coverage"] = b["mean_group_coverage"] - a["mean_group_coverage"]
    report["delta_full_coverage_rate"] = b["full_coverage_rate"] - a["full_coverage_rate"]
    # Only a byte-equivalent index control can be mechanically promoted using
    # this narrow suite alone. Changed builds also need reviewed labels and an
    # explicit answer-quality review; do not pretend lexical recall proves it.
    same_rows = all(a["cases"][cid]["retrieved_chunk_ids"] == b["cases"][cid]["retrieved_chunk_ids"] for cid in a["cases"])
    report["identical_control"] = ib.same_index_content(builds[0], builds[1], root)
    report["promotion_eligible"] = report["identical_control"] and same_rows and not report["regressions"] and b["mapping"]["all_groups_covered"]
    report["gate_explanation"] = ("Identical-index control only; not evidence that the baseline's answers are correct."
        if report["promotion_eligible"] else "Not auto-qualified. Inspect regressions; changed builds need source-label and end-to-end answer review.")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--root", type=Path, default=ib.BUILDS)
    p.add_argument("--url", default="http://localhost:6333")
    p.add_argument("--excerpts", default="eval/excerpts_v2.json")
    p.add_argument("--cases", default="eval/retry_eval_v1/dev_eval.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=24)
    p.add_argument("--token-budget", type=int, default=4000)
    p.add_argument("--allow-provider", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--sparse-only", action="store_true", help="BM25 diagnostic/control, not production hybrid quality")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("output exists; choose a new report name")
    load_dotenv(ROOT / ".env")
    with closing(qs.connect(a.url)) as client:
        report = evaluate(client, a.active, a.candidate, a.excerpts, a.cases,
                          k=a.k, candidates=a.candidates, token_budget=a.token_budget,
                          allow_provider=a.allow_provider, use_reranker=not a.no_rerank,
                          sparse_only=a.sparse_only, root=a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ib.write_once(a.out, report)
    print(f"Wrote {a.out}; promotion_eligible={report['promotion_eligible']}")
