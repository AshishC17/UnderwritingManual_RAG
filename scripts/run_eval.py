"""Evaluate retrieval configurations against an eval split.

    python scripts/run_eval.py                    # dev, full per-case detail
    python scripts/run_eval.py --split holdout    # aggregates only, by design

HOLDOUT POLICY — enforced here, not merely documented:

Running holdout is fine; that is what it exists for. *Reading individual holdout
failures* is not. The holdout number is a verdict, not a diagnostic — the moment
you tune against specific holdout cases it becomes a second dev set and you have
lost your only honest estimate of generalization.

So holdout prints aggregates only. Per-case detail, missed groups, and failure
listings are suppressed. Tune on dev; run holdout rarely, at milestones, to
confirm dev gains transferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.eval.harness import CONFIGS, prepare_queries, retrieve
from src.eval.metrics import aggregate, by_difficulty, percentile, score_case
from src.store import qdrant_store as qs

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SPLITS = {
    "dev": "eval/dev_eval_v2.json",
    "holdout": "eval/holdout_eval_v2.json",
    "all": "eval/ground_truth_v2.json",
    "dev_v1": "eval/dev_eval_v1.json",
    "holdout_v1": "eval/holdout_eval_v1.json",
}
# Only "dev" and "dev_v1" expose per-case detail; everything else is a verdict.
DIAGNOSABLE = {"dev", "dev_v1"}


def _rerank_cached(case: dict, client, dv, sv, args) -> bool:
    """True if this case's rerank is already cached, so no wait is needed."""
    from src.rerank.reranker import CACHE_DIR, MODEL, _key

    hits = qs.search_hybrid(
        client, dv, sv, limit=args.candidates, prefetch_limit=args.candidates * 2
    )
    texts = [h.payload["text"] for h in hits]
    return (Path(CACHE_DIR) / f"{_key(MODEL, case['question'], texts)}.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="dev")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--candidates", type=int, default=20,
                    help="shortlist size fed to the reranker")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--spacing", type=int, default=25,
                    help="seconds between uncached API calls (free tier: 3/min)")
    ap.add_argument("--out", default=None, help="write raw per-case results to JSON")
    args = ap.parse_args()

    protected = args.split not in DIAGNOSABLE

    data = json.loads((ROOT / SPLITS[args.split]).read_text())
    cases = data["cases"]

    print(f"split: {args.split}  cases: {len(cases)}  configs: {', '.join(args.configs)}")
    if protected:
        print("HOLDOUT MODE — aggregates only; per-case detail suppressed by policy.")
    print()

    print("preparing query embeddings (shared across all configs)")
    vectors = prepare_queries(cases, spacing_s=args.spacing)
    print()

    client = qs.connect()
    max_k = max(args.k)
    results: dict = {}

    completed: list[str] = []
    for config in args.configs:
        needs_api = config == "hybrid_rerank"
        print(f"running {config}{' (reranking — rate limited)' if needs_api else ''}")
        retrievals, latencies = {}, []

        try:
            for i, case in enumerate(cases):
                dv, sv = vectors[case["id"]]
                # Sleep before *every* uncached rerank, including the first: the
                # query-embedding pass just consumed this rate-limit window.
                if needs_api and not _rerank_cached(case, client, dv, sv, args):
                    import time as _t
                    _t.sleep(args.spacing)
                r = retrieve(
                    client, config, dv, sv,
                    k=max_k, candidates=args.candidates, query=case["question"],
                )
                retrievals[case["id"]] = r.chunk_ids
                latencies.append(r.latency_ms)
        except Exception as e:  # keep whatever already succeeded
            print(f"  {config} FAILED after {len(retrievals)}/{len(cases)} cases: "
                  f"{type(e).__name__}: {str(e)[:110]}")
            print(f"  reporting the {len(completed)} config(s) that completed.")
            continue

        completed.append(config)
        results[config] = {
            "retrievals": retrievals,
            "latency_p50": percentile(latencies, 50),
            "latency_p95": percentile(latencies, 95),
            "scores": {
                k: [score_case(c, retrievals[c["id"]], k) for c in cases]
                for k in args.k
            },
        }
    args.configs = completed
    if not completed:
        raise SystemExit("no configuration completed")
    print()

    # ---- headline table -------------------------------------------------
    print("=" * 84)
    print(f"RESULTS — {args.split}")
    print("=" * 84)
    header = f"{'config':<14} {'k':>3}  {'grp-recall':>10} {'full-cov':>9} {'MRR':>6} {'prec':>6}"
    print(header)
    print("-" * 84)
    for config in args.configs:
        for k in args.k:
            a = aggregate(results[config]["scores"][k])
            print(f"{config:<14} {k:>3}  {a['group_recall']:>10.3f} "
                  f"{a['full_coverage']:>9.3f} {a['mrr']:>6.3f} {a['precision']:>6.3f}")
        print("-" * 84)

    print(f"\n{'config':<14} {'p50 ms':>8} {'p95 ms':>8}   (retrieval only; "
          f"add ~400ms for query embedding in production)")
    for config in args.configs:
        print(f"{config:<14} {results[config]['latency_p50']:>8.0f} "
              f"{results[config]['latency_p95']:>8.0f}")

    # ---- per-difficulty -------------------------------------------------
    focus_k = args.k[min(1, len(args.k) - 1)]
    print(f"\nfull-coverage@{focus_k} by difficulty")
    diffs = ["easy", "medium", "difficult", "extreme"]
    print(f"{'config':<14} " + "".join(f"{d:>11}" for d in diffs))
    for config in args.configs:
        row = by_difficulty(results[config]["scores"][focus_k])
        cells = "".join(
            f"{row[d]['full_coverage']:>11.2f}" if d in row else f"{'-':>11}"
            for d in diffs
        )
        print(f"{config:<14} " + cells)

    # ---- stage disambiguation -------------------------------------------
    staged = [s for s in results[args.configs[0]]["scores"][focus_k]
              if s.stage_correct is not None]
    if staged:
        print(f"\nstage disambiguation @{focus_k} "
              f"({len(staged)} cases test it)")
        print(f"{'config':<14} {'accuracy':>9}   ranks (correct vs distractor)")
        for config in args.configs:
            ss = [s for s in results[config]["scores"][focus_k]
                  if s.stage_correct is not None]
            acc = sum(s.stage_correct for s in ss) / len(ss)
            detail = "  ".join(
                f"{s.case_id}:{s.stage_rank or '-'}v{s.distractor_rank or '-'}"
                for s in ss
            )
            print(f"{config:<14} {acc:>9.3f}   {detail}")

    # ---- per-case failures: DEV ONLY ------------------------------------
    if protected:
        print(f"\nPer-case detail withheld for {args.split}. Diagnose on dev.")
    else:
        print(f"\nunsatisfied evidence groups @{focus_k} (dev — diagnose these)")
        for config in args.configs:
            misses = [s for s in results[config]["scores"][focus_k] if not s.full_coverage]
            print(f"\n  {config}: {len(misses)} of {len(cases)} cases incomplete")
            for s in misses:
                print(f"    {s.case_id} [{s.difficulty:<9}] {s.groups_hit}/{s.groups_total} "
                      f"groups | missed: {', '.join(s.missed_groups)[:52]}")
                print(f"      ({s.stress_type})")

    if args.out:
        payload = {
            "split": args.split,
            "k_values": args.k,
            "configs": {
                c: {
                    "latency_p50": results[c]["latency_p50"],
                    "latency_p95": results[c]["latency_p95"],
                    "aggregate": {k: aggregate(results[c]["scores"][k]) for k in args.k},
                    "retrievals": results[c]["retrievals"],
                }
                for c in args.configs
            },
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nraw results -> {args.out}")


if __name__ == "__main__":
    main()
