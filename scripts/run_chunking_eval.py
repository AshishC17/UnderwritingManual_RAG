#!/usr/bin/env python3
"""Retrieval-only excerpt coverage against the current chunk build (stage 3 of the
chunker/index-versioning plan; see eval/excerpts_v1.json). Dev split only.

No decomposition, no correction retry — the raw case question goes straight to
embed -> search -> rerank, so orchestration cannot mask what the chunker did.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.eval.excerpts import assess, identified_in, tokens  # noqa: E402
from src.eval.harness import prepare_queries  # noqa: E402
from src.rerank.reranker import rerank  # noqa: E402
from src.store import qdrant_store as qs  # noqa: E402

CANDIDATES = 24  # matches MAX_RERANK_CANDIDATES in src/graph/pipeline.py


def retrieve_scoped(client, query: str, dense_vec, sparse_vec, k: int, candidates: int, query_filter) -> list[dict]:
    """hybrid_rerank, scoped to expected_scope -- the production retrieval config."""
    hits = qs.search_hybrid(
        client, dense_vec, sparse_vec, limit=candidates,
        prefetch_limit=candidates * 2, query_filter=query_filter,
    )
    ranked = rerank(query, [h.payload for h in hits], top_k=k)
    return [chunk for chunk, _ in ranked]


def excerpt_result(item: dict, retrieved: list[dict]) -> dict:
    if "source_scope" in item or "required_headers" in item:
        from src.eval.excerpts import evidence_result
        return evidence_result(item, retrieved)
    text, context = item["text"], item.get("context")
    own = [c for c in retrieved if c["source_doc"] == item["document"]]
    base = assess(text, own)
    resolved = bool(context is None or identified_in(text, context, own))
    status = base.status if (base.status != "intact" or resolved) else "wrong_context"
    return {"status": status, "coverage": base.coverage, "resolved": resolved,
            "chunk_ids": sorted({cid for cid in base.parts})}


def alternative_covered(alt: dict, excerpt_results: dict) -> tuple[bool, list[str]]:
    chunk_ids: set[str] = set()
    for eid in alt["excerpts"]:
        r = excerpt_results[eid]
        if r["status"] not in ("intact", "split") or not r["resolved"]:
            return False, []
        chunk_ids |= set(r["chunk_ids"])
    return True, sorted(chunk_ids)


def score_case(case: dict, excerpts: dict, groups: list[dict], retrieved: list[dict]) -> dict:
    excerpt_results = {eid: excerpt_result(excerpts[eid], retrieved) for eid in
                        {eid for g in groups for alt in g["any_of"] for eid in alt["excerpts"]}}
    group_results = []
    for g in groups:
        best = None
        for alt in g["any_of"]:
            covered, chunk_ids = alternative_covered(alt, excerpt_results)
            if covered and (best is None or len(chunk_ids) < len(best)):
                best = chunk_ids
        group_results.append({"group": g["group"], "covered": best is not None, "chunk_ids": best or []})
    full_coverage = all(g["covered"] for g in group_results)
    chunks_needed = len({cid for g in group_results for cid in g["chunk_ids"]}) if full_coverage else None
    split = sum(r["status"] == "split" for r in excerpt_results.values())
    found = sum(r["status"] in ("intact", "split") for r in excerpt_results.values())
    return {
        "groups": group_results, "full_coverage": full_coverage, "chunks_needed": chunks_needed,
        "group_coverage": sum(g["covered"] for g in group_results) / len(group_results),
        "split_rate": (split / found) if found else None,
        "excerpts": excerpt_results,
    }


def aggregate(per_case: dict, cases_by_id: dict) -> dict:
    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {"cases": 0}
        chunks_needed = [r["chunks_needed"] for r in rows if r["chunks_needed"] is not None]
        split_rates = [r["split_rate"] for r in rows if r["split_rate"] is not None]
        return {
            "cases": len(rows),
            "full_coverage": sum(r["full_coverage"] for r in rows) / len(rows),
            "group_coverage": sum(r["group_coverage"] for r in rows) / len(rows),
            "median_chunks_needed": sorted(chunks_needed)[len(chunks_needed) // 2] if chunks_needed else None,
            "split_rate": sum(split_rates) / len(split_rates) if split_rates else None,
        }

    rows = list(per_case.values())
    by_difficulty: dict[str, list[dict]] = {}
    for cid, r in per_case.items():
        by_difficulty.setdefault(cases_by_id[cid]["difficulty"], []).append(r)
    return {"overall": summarize(rows), "by_difficulty": {d: summarize(rs) for d, rs in by_difficulty.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpts", default="eval/excerpts_v1.json")
    ap.add_argument("--eval", default="eval/retry_eval_v1/dev_eval.json")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--candidates", type=int, default=CANDIDATES)
    ap.add_argument("--spacing", type=int, default=25)
    ap.add_argument("--out", default="results/chunking_eval_b0_dev.json")
    ap.add_argument("--label", default="b0_current_chunker", help="build id for this run")
    args = ap.parse_args()

    excerpt_doc = json.loads((ROOT / args.excerpts).read_text())
    excerpts, group_specs = excerpt_doc["excerpts"], excerpt_doc["groups"]
    eval_doc = json.loads((ROOT / args.eval).read_text())
    all_cases = {c["id"]: c for c in eval_doc.get("cases", eval_doc)}
    groups_by_case: dict[str, list[dict]] = {}
    for g in group_specs:
        groups_by_case.setdefault(g["case"], []).append(g)
    case_ids = [cid for cid in groups_by_case if all_cases[cid].get("expected_scope")]
    cases = [all_cases[cid] for cid in case_ids]
    print(f"build: {args.label}  cases: {len(cases)}  k: {args.k}\n")

    print("preparing query embeddings")
    vectors = prepare_queries(cases, spacing_s=args.spacing)
    print()

    client = qs.connect()
    max_k = max(args.k)
    per_k: dict[int, dict[str, dict]] = {k: {} for k in args.k}
    for i, case in enumerate(cases):
        scope = case["expected_scope"]
        query_filter = qs.scope_filter(lender_id=scope["lender_id"], document_version=scope["document_version"])
        dv, sv = vectors[case["id"]]
        print(f"  [{i + 1}/{len(cases)}] {case['id']} retrieving (rerank, rate limited)", flush=True)
        time.sleep(args.spacing)
        retrieved = retrieve_scoped(client, case["question"], dv, sv, max_k, args.candidates, query_filter)
        for k in args.k:
            per_k[k][case["id"]] = score_case(case, excerpts, groups_by_case[case["id"]], retrieved[:k])

    results = {
        "build": args.label, "split": "dev", "excerpts_file": args.excerpts,
        "excerpts_schema_version": excerpt_doc["schema_version"],
        "chunks_sha256": excerpt_doc["corpus_snapshot"]["chunks_sha256"],
        "config": "hybrid_rerank_scoped", "candidates": args.candidates,
        "by_k": {str(k): {"aggregate": aggregate(per_k[k], all_cases), "cases": per_k[k]} for k in args.k},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'k':>3} {'full-cov':>9} {'grp-cov':>8} {'med chunks':>11} {'split rate':>11}")
    for k in args.k:
        agg = results["by_k"][str(k)]["aggregate"]["overall"]
        print(f"{k:3d} {agg['full_coverage']:9.3f} {agg['group_coverage']:8.3f} "
              f"{str(agg['median_chunks_needed']):>11} {('-' if agg['split_rate'] is None else f'{agg['split_rate']:.3f}'):>11}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
