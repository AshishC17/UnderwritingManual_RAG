"""Run eval/conversation_eval_v1 in fixture mode and score each step.

    python scripts/run_conversation_eval.py --split dev
    python scripts/run_conversation_eval.py --split dev --case C-D01 C-D02
    python scripts/run_conversation_eval.py --split dev --out results/conversation_dev.json

Fixture mode only, per the suite's own staged rollout (fixture dev -> closed-loop
dev -> freeze -> holdout). Closed-loop mode — letting the system generate its own
setup answers instead of preloading verified/deliberately-incorrect fixtures — is
a separate block.

Scope, stated plainly rather than silently narrowed:

  SCORED today via score_generation (revised judge calibration pending):
    claim recall, hallucination (forbidden-claim assertion), citation validity,
    groundedness, abstention, relevance.
  SCORED today, mechanically, no model call:
    scope_pass (final scope vs expected.scope); unnecessary_lookup (now read
    directly off `evidence_reused` in graph state, since the graph has a real
    retrieval-skip path -- this no longer has to be inferred from whether
    sub_queries happened to be empty); behavior_pass (exact match against
    `expected.behavior`, since the graph now predicts an operation instead of
    always taking one path).
  NOT scored: format_pass, effective_context_group_coverage,
    retrieval_group_recall_at_k. Recorded as null with a reason rather than a
    fabricated pass/fail.

Every scored step writes one record shaped like result_record.schema.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.eval.generation_metrics import score_generation, format_metric
from src.eval.judge import JUDGE_MODEL, check_relevance, check_supported_batch, decompose_claims, judge_claim, classify_response, rubric_hashes
from src.graph.pipeline import build_graph
from src.graph.runtime import run_turn, seed_thread
from src.ingest.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EVAL_DIR = ROOT / "eval" / "conversation_eval_v1"
SPLITS = {"dev": "dev_eval.json", "holdout": "holdout_eval.json"}
DIAGNOSABLE = {"dev"}  # same policy as scripts/run_eval.py: holdout is a verdict


def load_split(split: str) -> dict:
    return json.loads((EVAL_DIR / SPLITS[split]).read_text())


def source_ref_map() -> dict[str, str]:
    """`source_ref -> chunk_id`, so `any_of_source_refs` can reuse the same
    evidence-group scoring the single-question eval already uses."""
    sources = json.loads((EVAL_DIR / "sources.json").read_text())["sources"]
    return {ref: entry["chunk_id"] for ref, entry in sources.items()}


def translate_groups(expected_groups: list[dict], ref_map: dict[str, str]) -> list[dict]:
    return [
        {"name": g["name"], "any_of_chunk_ids": [ref_map[r] for r in g["any_of_source_refs"]]}
        for g in expected_groups
    ]


def lender_names() -> dict[str, str]:
    specs = load_manifest("config/corpus_manifest.json", require_files=False)
    return {s.lender_id: s.lender_name for s in specs}


def fixture_history(
    fixtures: dict, fixture_id: str, names: dict[str, str], ref_map: dict[str, str]
) -> tuple[list[dict], dict | None]:
    """A fixture's `history` turns, reduced to what production actually threads
    through state: question, answer, and now the chunk ids the reuse path
    re-hydrates by id. `context_source_refs` are translated through
    `sources.json`, the same map `translate_groups` uses, so a fixture's
    evidence is exactly what `reuse_evidence_node` would have stored had this
    conversation happened for real. The last turn's scope becomes the seeded
    `prior_scope`."""
    fixture = fixtures[fixture_id]
    history = [
        {
            "question": t["question"],
            "answer": t["answer"],
            "chunk_ids": [ref_map[r] for r in t.get("context_source_refs", [])],
        }
        for t in fixture["history"]
    ]
    last_scope = fixture["history"][-1].get("scope")
    scope = (
        {
            "status": "resolved",
            "lender_id": last_scope["lender_id"],
            "lender_name": names[last_scope["lender_id"]],
            "document_version": last_scope["document_version"],
        }
        if last_scope
        else None
    )
    return history, scope


def score_step(case: dict, step: dict, final: dict, ref_map: dict[str, str], judge_model: str) -> dict:
    expected = step["expected"]
    interrupted = bool(final.get("__interrupt__"))
    # Evaluate the original request in its conversation, not only a model rewrite
    # that may have dropped a condition. Prior assistant text is not policy truth.
    grading_input = json.dumps({
        "current_user_message": step["user_message"],
        "prior_conversation": final.get("history") or [],
    }, ensure_ascii=False)

    pseudo_case = {
        "id": f"{case['id']}:{step['step_id']}",
        "question": grading_input,
        "difficulty": case["difficulty"],
        "stress_type": case["scenario_family"],
        "required_claims": expected["required_claims"],
        "forbidden_claims": expected["forbidden_claims"],
        "evidence_groups": translate_groups(expected.get("evidence_groups", []), ref_map),
    }

    answer = final.get("answer") or ""
    context_ids = [c["chunk_id"] for c in (final.get("context_chunks") or [])]
    response_assessment = None if interrupted else classify_response(grading_input, answer, model=judge_model)

    gen_score = None
    if not interrupted and pseudo_case["required_claims"]:
        gen_score = score_generation(
            pseudo_case, answer, context_ids,
            judge=lambda claim, ans: judge_claim(claim, ans, model=judge_model),
            context="\n\n".join(c.get("text", "") for c in (final.get("context_chunks") or [])),
            decompose=lambda a: decompose_claims(a, model=judge_model),
            supported=lambda cs, ctx: check_supported_batch(cs, ctx, model=judge_model, question=grading_input),
            relevance=lambda q, a: check_relevance(q, a, model=judge_model),
            response_assessment=response_assessment,
        )

    # scope_pass: mechanical, no model call.
    scope_pass = None
    exp_scope = expected.get("scope")
    if expected["behavior"] == "clarify":
        scope_pass = interrupted
    elif exp_scope is not None and not interrupted:
        actual = final.get("scope") or {}
        scope_pass = (
            actual.get("lender_id") == exp_scope["lender_id"]
            and actual.get("document_version") == exp_scope["document_version"]
        )

    # unnecessary_lookup: read directly off graph state now, not inferred.
    # `evidence_reused` is set exactly once, by `reuse_evidence_node`, and
    # reset to False at the start of every turn -- it is a fact about what
    # ran this turn, not a guess from a side effect like sub_queries being
    # non-empty.
    unnecessary_lookup = None
    if expected["evidence_policy"] == "stored_evidence_sufficient" and not interrupted:
        unnecessary_lookup = not bool(final.get("evidence_reused"))

    predicted_operation = final.get("behavior") if not interrupted else "clarify"
    behavior_pass = (
        (predicted_operation == expected["behavior"]) if predicted_operation else None
    )

    return {
        "run_id": case["_run_id"],
        "case_id": case["id"],
        "step_id": step["step_id"],
        "mode": "fixture",
        "thread_id": case["_thread_ids"][step["thread"]],
        "status": "interrupted" if interrupted else "completed",
        "user_message": step["user_message"],
        "resolved_question": final.get("resolved_question"),
        "predicted_operation": predicted_operation,
        "scope": final.get("scope"),
        "answer": answer if not interrupted else None,
        "response_assessment": response_assessment,
        "metrics": {
            "claim_recall": gen_score.claim_recall if gen_score else None,
            "groundedness": gen_score.groundedness if gen_score else None,
            "forbidden_claim_present": bool(gen_score.hallucinated) if gen_score else None,
            "citation_validity": gen_score.citation_validity if gen_score else None,
            "behavior_pass": behavior_pass,
            "scope_pass": scope_pass,
            "format_pass": None,  # not yet implemented
            "unnecessary_lookup": unnecessary_lookup,
        },
        "not_applicable_reasons": {
            "format_pass": "output-format checking not yet implemented",
        },
    }


def run_case(
    app, case: dict, fixtures: dict, run_id: str, ref_map: dict[str, str],
    judge_model: str, names: dict[str, str],
) -> list[dict]:
    case["_run_id"] = run_id
    case["_thread_ids"] = {
        key: f"{run_id}:{case['id']}:{key}" for key in {s["thread"] for s in case["steps"]}
    }

    for thread_key, fixture_id in case.get("fixture_sessions", {}).items():
        history, scope = fixture_history(fixtures, fixture_id, names, ref_map)
        seed_thread(app, case["_thread_ids"][thread_key], history, scope)

    records = []
    for step in case["steps"]:
        thread_id = case["_thread_ids"][step["thread"]]
        final = run_turn(app, thread_id, step["user_message"])
        records.append(score_step(case, step, final, ref_map, judge_model))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="dev")
    ap.add_argument("--case", nargs="+", default=None, help="run only these case IDs")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    protected = args.split not in DIAGNOSABLE
    data = load_split(args.split)
    cases = data["cases"]
    if args.case:
        by_id = {c["id"]: c for c in cases}
        missing = [c for c in args.case if c not in by_id]
        if missing:
            raise SystemExit(f"unknown case ID(s): {', '.join(missing)}")
        cases = [by_id[c] for c in args.case]

    ref_map = source_ref_map()
    names = lender_names()
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    app = build_graph()

    print(f"split: {args.split}  cases: {len(cases)}  run_id: {run_id}")
    if protected:
        print("HOLDOUT MODE — aggregates only; per-case detail suppressed by policy.")
    print()

    all_records = []
    partial = False
    for i, case in enumerate(cases, 1):
        t0 = time.monotonic()
        try:
            records = run_case(app, case, data["fixtures"], run_id, ref_map, args.judge_model, names)
        except Exception as e:
            print(f"\n  STOPPED at {case['id']}: {type(e).__name__}: {str(e)[:160]}")
            print(f"  reporting the {len(all_records)} step(s) that completed.")
            partial = True
            break
        all_records.extend(records)
        elapsed = time.monotonic() - t0
        if protected:
            print(f"  [{i}/{len(cases)}] scored", flush=True)
        else:
            summary = ", ".join(
                f"{r['step_id']}:{r['status']}"
                + (f" claim_recall={r['metrics']['claim_recall']:.2f}" if r["metrics"]["claim_recall"] is not None else "")
                for r in records
            )
            print(f"  [{i}/{len(cases)}] {case['id']:<7} {elapsed:>5.1f}s  {summary}")

    if not all_records:
        raise SystemExit("no case completed")

    scored = [r for r in all_records if r["metrics"]["claim_recall"] is not None]
    lookup_checked = [r for r in all_records if r["metrics"]["unnecessary_lookup"] is not None]
    scope_checked = [r for r in all_records if r["metrics"]["scope_pass"] is not None]
    behavior_checked = [r for r in all_records if r["metrics"]["behavior_pass"] is not None]

    print("\n" + "=" * 72)
    print(f"CONVERSATION EVAL — {args.split} ({len(all_records)} steps, fixture mode)")
    print("=" * 72)
    if scored:
        def avg(key):
            values = [r["metrics"][key] for r in scored if r["metrics"][key] is not None]
            return sum(values) / len(values) if values else None
        print(f"  claim recall           {avg('claim_recall'):.3f}   (n={len(scored)})")
        print(f"  groundedness            {format_metric(avg('groundedness'))}")
        halluc = sum(r["metrics"]["forbidden_claim_present"] for r in scored) / len(scored)
        print(f"  hallucination rate      {halluc:.3f}")
        cite = avg("citation_validity")
        print(f"  citation validity       {format_metric(cite)}")
    if scope_checked:
        sp = sum(r["metrics"]["scope_pass"] for r in scope_checked) / len(scope_checked)
        print(f"  scope correctness       {sp:.3f}   (n={len(scope_checked)})")
    if lookup_checked:
        ul = sum(r["metrics"]["unnecessary_lookup"] for r in lookup_checked) / len(lookup_checked)
        print(f"  unnecessary lookup rate {ul:.3f}   (n={len(lookup_checked)}, stored_evidence_sufficient steps)")
    if behavior_checked:
        bp = sum(r["metrics"]["behavior_pass"] for r in behavior_checked) / len(behavior_checked)
        print(f"  behavior correctness    {bp:.3f}   (n={len(behavior_checked)})")
        if not protected:
            for r in behavior_checked:
                if not r["metrics"]["behavior_pass"]:
                    print(f"     mismatch {r['case_id']}:{r['step_id']}  "
                          f"predicted={r['predicted_operation']}")
    print(f"\n  format_pass: not scored — see module docstring")

    if not protected:
        bad = [r for r in scored if r["metrics"]["forbidden_claim_present"]]
        if bad:
            print(f"\nhallucinated steps ({len(bad)}):")
            for r in bad:
                print(f"  {r['case_id']}:{r['step_id']}  \"{r['user_message'][:60]}\"")

    out_path = Path(args.out) if args.out else ROOT / "results" / f"conversation_{args.split}.json"
    if partial:
        out_path = out_path.with_suffix(".partial.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_id": run_id,
        "split": args.split,
        "mode": "fixture",
        "judge_model": args.judge_model,
        "evaluator_rubric_hashes": rubric_hashes(),
        "planned_steps": sum(len(c["steps"]) for c in cases),
        "recorded_steps": len(all_records),
        "metric_denominators": {
            key: sum(r["metrics"][key] is not None for r in all_records)
            for key in all_records[0]["metrics"]
        },
        "records": all_records,
    }, indent=2))
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
