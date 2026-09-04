"""Hand-check the LLM judge, then score the agreement properly.

    python scripts/validate_judge.py results/judge_audit_dev.json --sample 40
    python scripts/validate_judge.py results/judge_audit_dev.json --score

The first form prints a numbered sample for you to check and writes a template
for your labels. The second reads your labels back and reports Cohen's kappa and
per-class recall.

Why not raw agreement: forbidden claims are mostly genuinely absent, so "absent"
is the majority class. A judge that answered "absent" to everything would score
~90% agreement while catching zero hallucinations — the kappa prevalence paradox.
Kappa corrects for chance; per-class recall shows what is actually disagreeing.
Published target for judge-human kappa is 0.7-0.8, the human-human ceiling.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _answers_by_case(split: str) -> dict[str, tuple[str, str]]:
    """Reconstruct (question, full_answer) for every case in a split.

    Re-runs retrieval and generation, but every step — dense/sparse search,
    rerank, and the generation call itself — reads from its own disk cache with
    identical inputs, so this costs zero API calls. The audit file only stores
    the judge's per-claim verdict and its (possibly wrong) quoted evidence; you
    cannot tell whether a verdict is correct without the actual question and the
    full answer it was judged against.
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    from src.embed.embedder import embed_query
    from src.embed.sparse import embed_query as embed_query_sparse
    from src.generate.generator import generate
    from src.rerank.reranker import rerank
    from src.store import qdrant_store as qs

    path = Path(__file__).resolve().parents[1] / "eval" / f"{split}_eval_v2.json"
    cases = json.loads(path.read_text())["cases"]
    client = qs.connect()

    out: dict[str, tuple[str, str]] = {}
    for case in cases:
        q = case["question"]
        hits = qs.search_hybrid(
            client, embed_query(q), embed_query_sparse(q),
            limit=15, prefetch_limit=30,
        )
        chunks = [p for p, _ in rerank(q, [h.payload for h in hits], top_k=10)]
        out[case["id"]] = (q, generate(q, chunks))
    return out


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa over (judge, human) verdict pairs."""
    n = len(pairs)
    if not n:
        return 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    labels = {v for p in pairs for v in p}
    expected = sum(
        (sum(1 for a, _ in pairs if a == v) / n) * (sum(1 for _, b in pairs if b == v) / n)
        for v in labels
    )
    return 0.0 if expected >= 1.0 else (observed - expected) / (1 - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit", help="judge audit JSON from --export-judge")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default=None,
                    help="your labels file (default: <audit>.labels.json)")
    ap.add_argument("--score", action="store_true", help="score existing labels")
    ap.add_argument("--split", default="dev",
                    help="which eval split the audit came from, to look up "
                         "questions and re-derive full answers (cached, free)")
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    labels_path = Path(args.labels or (args.audit + ".labels.json"))

    # Sample both classes deliberately: a random draw would be almost all
    # "absent", and the verdicts worth checking are the "present" ones.
    rng = random.Random(args.seed)
    present = [r for r in audit if r["verdict"] == "present"]
    absent = [r for r in audit if r["verdict"] == "absent"]
    half = args.sample // 2
    picked = (rng.sample(present, min(half, len(present)))
              + rng.sample(absent, min(args.sample - half, len(absent))))
    rng.shuffle(picked)

    if not args.score:
        print(f"{len(audit)} verdicts total "
              f"({len(present)} present, {len(absent)} absent)")
        print(f"sampling {len(picked)} — deliberately balanced, not random,\n"
              f"because a random draw would be nearly all 'absent'.")
        print("re-deriving full questions and answers from cache "
              "(no API calls)...\n")
        answers = _answers_by_case(args.split)

        template = []
        for i, r in enumerate(picked, 1):
            question, full_answer = answers.get(r["case_id"], ("(unknown)", "(unknown)"))
            print("=" * 76)
            print(f"[{i}] case {r['case_id']}")
            print(f"  QUESTION: {question}")
            print(f"  FULL ANSWER:\n    " + full_answer.replace("\n", "\n    "))
            print(f"  ---")
            print(f"  CLAIM        : {r['claim']}")
            print(f"  JUDGE SAID   : {r['verdict'].upper()}")
            print(f"  JUDGE'S QUOTE: {r['evidence'] or '(none quoted)'}")
            template.append({
                "i": i, "case_id": r["case_id"], "question": question,
                "full_answer": full_answer, "claim": r["claim"],
                "judge": r["verdict"], "judge_evidence": r["evidence"], "human": "",
            })

        labels_path.write_text(json.dumps(template, indent=2))
        print("\n" + "=" * 76)
        print(f"Template written to {labels_path}")
        print('For each row, read QUESTION + FULL ANSWER yourself, decide whether\n'
              'the answer actually asserts the CLAIM, then fill "human" with\n'
              '"present" or "absent" — your own judgement, not the judge\'s quote.')
        print(f"Then: python {Path(__file__).name} {args.audit} --score")
        return

    rows = json.loads(labels_path.read_text())
    done = [r for r in rows if r.get("human") in {"present", "absent"}]
    if not done:
        raise SystemExit(f"no labels filled in {labels_path}")

    pairs = [(r["judge"], r["human"]) for r in done]
    agree = sum(1 for a, b in pairs if a == b) / len(pairs)
    k = kappa(pairs)

    # Per-class recall: of claims that truly ARE present, how many did the judge
    # catch? This is the number the majority class cannot inflate.
    truly_present = [r for r in done if r["human"] == "present"]
    caught = sum(1 for r in truly_present if r["judge"] == "present")
    truly_absent = [r for r in done if r["human"] == "absent"]
    correct_absent = sum(1 for r in truly_absent if r["judge"] == "absent")

    print(f"labels checked: {len(done)} of {len(rows)}")
    print(f"  raw agreement    {agree:.3f}   <- do not report this alone")
    print(f"  Cohen's kappa    {k:.3f}   "
          f"{'OK (>=0.7)' if k >= 0.7 else 'BELOW TARGET — tighten the judge prompt'}")
    if truly_present:
        print(f"  recall (present) {caught}/{len(truly_present)} = "
              f"{caught/len(truly_present):.3f}   <- the number that matters")
    if truly_absent:
        print(f"  recall (absent)  {correct_absent}/{len(truly_absent)} = "
              f"{correct_absent/len(truly_absent):.3f}")

    disagreements = [r for r in done if r["judge"] != r["human"]]
    if disagreements:
        print(f"\ndisagreements ({len(disagreements)}):")
        for r in disagreements:
            print(f"  {r['case_id']}  judge={r['judge']} human={r['human']}")
            print(f"     claim: {r['claim'][:88]}")


if __name__ == "__main__":
    main()
