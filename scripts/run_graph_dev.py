"""Run the scoped dev split through the LangGraph pipeline and score retrieval.

    python scripts/run_graph_dev.py

Traces land in the LangSmith project named below, kept separate from ad-hoc
runs so this experiment is findable later. Roughly 11 runs per question
(decompose + per-sub-query embed/search/rerank + generate), ~154 for the split
— about 3% of the 5,000/month free tier.

Slow by construction: Voyage's free tier is 3 requests/min, so a 3-sub-query
question paces to ~90s. Partial results are written on failure rather than
discarded, and every stage is cached, so a re-run resumes rather than restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Set before importing anything that builds a LangSmith client.
os.environ.setdefault("LANGSMITH_PROJECT", "rag-uw-decomposition-dev")

from src.eval.metrics import aggregate, by_difficulty, score_case  # noqa: E402
from src.graph.pipeline import (  # noqa: E402
    CANDIDATES_PER_SUB_QUERY,
    FINAL_K,
    GLOBAL_RERANK_TOKEN_BUDGET,
    MAX_RERANK_CANDIDATES,
    MAX_RERANK_PAIRS,
    build_graph,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--out", default="results/v3_graph_dev.json")
    ap.add_argument("--case", nargs="+", default=None,
                    help="run only these case IDs, in the supplied order")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N cases (for a quick check)")
    args = ap.parse_args()

    cases = json.loads(
        (ROOT / f"eval/{args.split}_eval_v2.json").read_text()
    )["cases"]
    if args.case:
        by_id = {case["id"]: case for case in cases}
        missing = [case_id for case_id in args.case if case_id not in by_id]
        if missing:
            raise SystemExit(f"unknown case ID(s): {', '.join(missing)}")
        cases = [by_id[case_id] for case_id in args.case]
    if args.limit:
        cases = cases[: args.limit]

    print(f"split: {args.split}  cases: {len(cases)}")
    print(f"LangSmith project: {os.environ['LANGSMITH_PROJECT']}")
    print(f"tracing enabled: {os.environ.get('LANGSMITH_TRACING')}\n")

    app = build_graph()
    retrievals: dict[str, list[str]] = {}
    scope_decisions: dict[str, dict] = {}
    metadata_filters: dict[str, dict | None] = {}
    sub_query_counts: dict[str, int] = {}
    rerank_pair_counts: dict[str, int] = {}
    partial = False

    needs_clarification: list[str] = []

    for i, case in enumerate(cases, 1):
        t0 = time.monotonic()
        try:
            # One thread per case: these are independent questions, and sharing
            # a thread would let one case's state leak into the next.
            final = app.invoke(
                {"question": case["question"]},
                config={"configurable": {"thread_id": f"{args.split}:{case['id']}"}},
            )
        except Exception as e:  # keep what completed; the caches preserve it
            print(f"\n  STOPPED at {case['id']}: {type(e).__name__}: {str(e)[:140]}")
            print(f"  reporting the {len(retrievals)} case(s) that completed.")
            partial = True
            break

        # The graph now pauses for a clarification instead of answering, so an
        # unscoped question yields no retrieval to score. Record and continue:
        # this is the resolver working, not a failure of the run.
        if final.get("__interrupt__"):
            needs_clarification.append(case["id"])
            scope_decisions[case["id"]] = final["scope"]
            metadata_filters[case["id"]] = None
            print(f"  [{i}/{len(cases)}] {case['id']:<5} needs clarification "
                  f"({final['scope'].get('reason')}) — not scored", flush=True)
            continue

        scope_decisions[case["id"]] = final["scope"]
        metadata_filters[case["id"]] = final["metadata_filter"]

        # Score against the full reranked order, not just the top-k handed to
        # the model, so recall@20 is measurable rather than capped at FINAL_K.
        ordered = [c["chunk_id"] for c, _ in final["reranked"]]
        retrievals[case["id"]] = ordered
        sub_query_counts[case["id"]] = len(final["sub_queries"])
        rerank_pair_counts[case["id"]] = len(final["rerank_assignments"])
        scope = final["scope"]
        scope_label = (
            f"{scope['lender_id']}:{scope['document_version']}"
            if scope["status"] == "resolved"
            else f"clarify:{scope['reason']}"
        )
        print(f"  [{i}/{len(cases)}] {case['id']:<5} {scope_label:<26} "
              f"{len(final['sub_queries'])} sub-q  "
              f"{len(final['candidates'])} unique / "
              f"{len(final['rerank_assignments'])} rerank pairs  "
              f"{time.monotonic() - t0:.0f}s", flush=True)

    if needs_clarification:
        print(f"\n{len(needs_clarification)} of {len(cases)} case(s) stopped for "
              f"clarification and were not scored: {', '.join(needs_clarification)}")
        print("  Questions that name no lender cannot be scoped. Add the lender "
              "and version to the question text to score them.")

    if not retrievals:
        raise SystemExit(
            "no case completed — every question needed clarification. "
            "The eval set predates the two-lender corpus."
        )

    scored_cases = [c for c in cases if c["id"] in retrievals]
    print("\n" + "=" * 72)
    print(f"GRAPH RETRIEVAL — {args.split} ({len(scored_cases)} cases)")
    print("=" * 72)
    print(f"{'k':>3}  {'grp-recall':>10} {'full-cov':>9} {'MRR':>6} {'prec':>6}")
    results = {}
    for k in args.k:
        scores = [score_case(c, retrievals[c["id"]], k) for c in scored_cases]
        a = aggregate(scores)
        results[k] = a
        print(f"{k:>3}  {a['group_recall']:>10.3f} {a['full_coverage']:>9.3f} "
              f"{a['mrr']:>6.3f} {a['precision']:>6.3f}")

    focus = args.k[min(1, len(args.k) - 1)]
    focus_scores = [score_case(c, retrievals[c["id"]], focus) for c in scored_cases]
    print(f"\nfull-coverage@{focus} by difficulty")
    for d, a in by_difficulty(focus_scores).items():
        print(f"  {d:<10} {a['full_coverage']:.3f}  ({a['cases']} cases)")

    print(f"\nunsatisfied evidence groups @{focus}")
    for s in focus_scores:
        if not s.full_coverage:
            print(f"  {s.case_id} [{s.difficulty:<9}] {s.groups_hit}/{s.groups_total} "
                  f"| missed: {', '.join(s.missed_groups)[:50]}")

    out = Path(args.out)
    if partial:
        out = out.with_suffix(".partial" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split,
        "pipeline": "langgraph resolve_scope->decompose->retrieve->rerank->generate",
        "pipeline_config": {
            "candidates_per_sub_query": CANDIDATES_PER_SUB_QUERY,
            "max_rerank_candidates": MAX_RERANK_CANDIDATES,
            "max_rerank_pairs": MAX_RERANK_PAIRS,
            "global_rerank_token_budget": GLOBAL_RERANK_TOKEN_BUDGET,
            "final_k": FINAL_K,
            "candidate_owner": "best_rrf_rank",
        },
        "langsmith_project": os.environ["LANGSMITH_PROJECT"],
        "scope_decisions": scope_decisions,
        "metadata_filters": metadata_filters,
        "sub_query_counts": sub_query_counts,
        "rerank_pair_counts": rerank_pair_counts,
        "aggregate": {str(k): v for k, v in results.items()},
        "retrievals": retrievals,
    }, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
