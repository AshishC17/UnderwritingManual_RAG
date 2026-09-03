"""Generate answers over retrieved context and score them.

    python scripts/run_generation_eval.py --split dev

Requires VOYAGE_API_KEY (retrieval) and GROQ_API_KEY (generation and judging).
Answers and judgements are cached, so re-runs are free.

Groq enforces a per-model daily token cap. Groundedness sends the context once
per answer rather than once per claim, which is roughly an order of magnitude
cheaper on a typical answer; a rate-limit failure still reports whatever scored
before it, rather than discarding the run.

Also writes a judge-validation file. The judge is unvalidated until a human
checks it, and an unvalidated judge silently corrupts every number here — so
`--export-judge` dumps every verdict with its quoted evidence for hand-checking.

The holdout guard applies as it does for retrieval: per-case detail on dev only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.embed.embedder import embed_query
from src.embed.sparse import embed_query as embed_query_sparse
from src.eval.generation_metrics import (
    aggregate_generation,
    by_difficulty_generation,
    score_generation,
)
from src.eval.judge import (
    JUDGE_MODEL,
    check_relevance,
    check_supported_batch,
    decompose_claims,
    judge_claim,
)
from src.generate.generator import MODEL as GEN_MODEL
from src.generate.generator import MissingCredentials, build_context, generate
from src.rerank.reranker import rerank
from src.store import qdrant_store as qs

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SPLITS = {
    "dev": "eval/dev_eval_v2.json",
    "holdout": "eval/holdout_eval_v2.json",
}
DIAGNOSABLE = {"dev"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="dev")
    ap.add_argument("--k", type=int, default=10, help="chunks passed to the model")
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--export-judge", default=None,
                    help="write every judge verdict + evidence for hand-checking")
    args = ap.parse_args()

    protected = args.split not in DIAGNOSABLE
    cases = json.loads((ROOT / SPLITS[args.split]).read_text())["cases"]

    print(f"split: {args.split}  cases: {len(cases)}  k: {args.k}")
    print(f"generator: {args.gen_model}   judge: {args.judge_model}")
    if args.gen_model == args.judge_model:
        print("WARNING: judge and generator are the same model — self-preference bias")
    if protected:
        print("HOLDOUT MODE — aggregates only; per-case detail suppressed by policy.")
    print()

    client = qs.connect()
    audit: list[dict] = []

    def make_judge(case_id: str):
        def _judge(claim: str, answer: str):
            present, evidence = judge_claim(claim, answer, model=args.judge_model)
            audit.append({
                "case_id": case_id, "claim": claim,
                "verdict": "present" if present else "absent",
                "evidence": evidence,
            })
            return present, evidence
        return _judge

    scores = []
    partial = False
    for i, case in enumerate(cases, 1):
        q = case["question"]
        hits = qs.search_hybrid(
            client, embed_query(q), embed_query_sparse(q),
            limit=args.candidates, prefetch_limit=args.candidates * 2,
        )
        ranked = rerank(q, [h.payload for h in hits], top_k=args.k)
        chunks = [p for p, _ in ranked]

        try:
            answer = generate(q, chunks, model=args.gen_model)
        except MissingCredentials as e:
            print(f"\nERROR: {e}")
            raise SystemExit(1)

        try:
            s = score_generation(
                case, answer, [c["chunk_id"] for c in chunks],
                make_judge(case["id"]),
                context=build_context(chunks),
                decompose=(lambda a: decompose_claims(a, model=args.judge_model)),
                supported=(lambda cs, ctx: check_supported_batch(
                    cs, ctx, model=args.judge_model)),
                relevance=(lambda q, a: check_relevance(q, a, model=args.judge_model)),
            )
        except Exception as e:  # keep what already scored; the cache preserves it
            print(f"\n  STOPPED at {case['id']}: {type(e).__name__}: {str(e)[:160]}")
            print(f"  reporting the {len(scores)} case(s) that completed. "
                  f"Cached work is preserved — re-run to continue.")
            partial = True
            break
        scores.append(s)
        # Progress only on a diagnosable split. On holdout even a per-case
        # pass/fail line is tuning signal, which the policy exists to withhold.
        if protected:
            print(f"  [{i}/{len(cases)}] scored", flush=True)
        else:
            print(f"  [{i}/{len(cases)}] {case['id']}  claims {s.claims_present}/"
                  f"{s.claims_required}  "
                  f"{'HALLUCINATED' if s.hallucinated else 'clean'}")

    if not scores:
        raise SystemExit('no case completed')

    agg = aggregate_generation(scores)
    print("\n" + "=" * 72)
    print(f"GENERATION RESULTS — {args.split}")
    print("=" * 72)
    print(f"  claim recall           {agg['claim_recall']:.3f}   "
          f"(required claims the answer asserts)")
    print(f"  groundedness           {agg['groundedness']:.3f}   "
          f"(claims it makes that the context supports)")
    print(f"  hallucination rate     {agg['hallucination_rate']:.3f}   "
          f"(questions asserting ANY forbidden claim)")
    print(f"  answer relevance       {agg['answer_relevance']:.3f}   "
          f"(answers that address the question)")
    print(f"  citation validity      {agg['citation_validity']:.3f}   "
          f"(cited ids that were actually retrieved)")
    print(f"  over-refusal           {agg['over_refusal']:.3f}   "
          f"(declined despite having the evidence)")
    print(f"  unsupported confidence {agg['unsupported_confidence']:.3f}   "
          f"(of {agg['cases_missing_evidence']} cases missing evidence, "
          f"answered anyway)")
    print(f"  ungrounded claims      {agg['ungrounded_claims']} total | "
          f"forbidden assertions {agg['forbidden_assertions']}")

    print("\nby difficulty")
    print(f"{'difficulty':<12} {'claim-recall':>13} {'halluc-rate':>12}")
    for d, a in by_difficulty_generation(scores).items():
        print(f"{d:<12} {a['claim_recall']:>13.3f} {a['hallucination_rate']:>12.3f}")

    if protected:
        print(f"\nPer-case detail withheld for {args.split}. Diagnose on dev.")
    else:
        bad = [s for s in scores if s.hallucinated]
        print(f"\nhallucinated cases ({len(bad)} of {len(scores)}) — diagnose these")
        for s in bad:
            print(f"  {s.case_id} [{s.difficulty}] ({s.stress_type})")
            for c in s.forbidden_asserted:
                print(f"      asserted forbidden: {c[:88]}")

        ung = [s for s in scores if s.ungrounded_claims]
        print(f"\nungrounded claims ({len(ung)} cases) — asserted without context support")
        for s in ung:
            print(f"  {s.case_id} [{s.difficulty}] "
                  f"{s.claims_grounded}/{s.claims_made} grounded")
            for c in s.ungrounded_claims[:3]:
                print(f"      {c[:92]}")

        risky = [s for s in scores if s.unsupported_confidence or s.over_refusal]
        if risky:
            print("\nabstention problems")
            for s in risky:
                kind = ("answered confidently without complete evidence"
                        if s.unsupported_confidence else "declined despite having evidence")
                print(f"  {s.case_id} [{s.difficulty}] — {kind}")

    def _dest(path: str) -> Path:
        """A partial run writes to *.partial.json — overwriting a complete
        result with a truncated one silently destroys measured work."""
        p = Path(path)
        return p.with_suffix(".partial" + p.suffix) if partial else p

    if args.export_judge:
        dest = _dest(args.export_judge)
        dest.write_text(json.dumps(audit, indent=2))
        print(f"\njudge verdicts -> {dest}  ({len(audit)} to hand-check)")
    if args.out:
        _dest(args.out).write_text(json.dumps({
            "split": args.split, "k": args.k,
            "gen_model": args.gen_model, "judge_model": args.judge_model,
            "aggregate": agg,
            "by_difficulty": by_difficulty_generation(scores),
            "cases": [vars(s) for s in scores],
        }, indent=2))
        print(f"results -> {_dest(args.out)}")


if __name__ == "__main__":
    main()
