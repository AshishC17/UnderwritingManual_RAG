"""Run the dedicated cross-document comparison evaluation.

Examples:
  .venv/bin/python scripts/run_comparison_eval.py --split dev --skip-judge
  .venv/bin/python scripts/run_comparison_eval.py --split dev --case C-D01
  .venv/bin/python scripts/run_comparison_eval.py --split holdout

The graph never receives ground-truth claims or evidence IDs. Mechanical
orchestration/retrieval metrics are computed after the turn. Unless
``--skip-judge`` is used, generation quality is scored separately by the
configured Qwen judge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.eval.comparison_metrics import aggregate_comparison, score_comparison_execution
from src.eval.generation_metrics import aggregate_generation, score_generation
from src.eval.judge import (
    EVALUATOR_VERSION,
    JUDGE_MODEL,
    check_relevance,
    check_supported_batch,
    classify_response,
    decompose_claims,
    judge_claim,
    rubric_hashes,
)
from src.generate.generator import build_comparison_context
from src.graph.pipeline import build_graph
from src.graph.runtime import run_turn, seed_thread
from src.ingest.manifest import load_manifest
from src.resolve.scope import scope_key

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval" / "comparison_eval_v1"
FILES = {"dev": "dev_eval.json", "holdout": "holdout_eval.json"}
DEFAULT_CORRECTION_MODE = "post_generation"
DEFAULT_MAX_RETRIES = 1
load_dotenv(ROOT / ".env")


def _scope_catalog() -> dict[tuple[str, str], dict]:
    result = {}
    for spec in load_manifest("config/corpus_manifest.json", require_files=False):
        result[(spec.lender_id, spec.document_version)] = {
            "status": "resolved",
            "lender_id": spec.lender_id,
            "lender_name": spec.lender_name,
            "document_id": spec.document_id,
            "document_version": spec.document_version,
            "effective_from": spec.effective_from,
            "effective_to": spec.effective_to,
        }
    return result


def _seed_setup(graph, thread_id: str, setup: dict, catalog: dict) -> None:
    prior = setup.get("prior_scope")
    runtime_prior = (
        catalog[(prior["lender_id"], prior["document_version"])] if prior else None
    )
    history = []
    for turn in setup.get("history") or []:
        scope = turn["scope"]
        history.append({
            **turn,
            "scope": catalog[(scope["lender_id"], scope["document_version"])],
        })
    if history or runtime_prior:
        seed_thread(graph, thread_id, history, runtime_prior)


def _generation_score(case: dict, final: dict, judge_model: str) -> dict | None:
    if (final.get("__interrupt__") or final.get("guard_reason")
            or not case["expected"]["required_claims"]):
        return None
    expected = case["expected"]
    contexts = final.get("comparison_context") or {}
    retrieved_ids = [
        chunk["chunk_id"] for chunks in contexts.values() for chunk in chunks
    ]
    pseudo_case = {
        "id": case["id"],
        "question": case["question"],
        "difficulty": case["difficulty"],
        "stress_type": case["family"],
        "required_claims": expected["required_claims"],
        "forbidden_claims": expected["forbidden_claims"],
        "evidence_groups": expected["evidence_groups"],
    }
    answer = final.get("answer") or ""
    assessment = classify_response(case["question"], answer, model=judge_model)
    score = score_generation(
        pseudo_case,
        answer,
        retrieved_ids,
        judge=lambda claim, value: judge_claim(claim, value, model=judge_model),
        context=build_comparison_context(final.get("comparison_scopes") or [], contexts),
        decompose=lambda value: decompose_claims(value, model=judge_model),
        supported=lambda claims, context: check_supported_batch(
            claims, context, model=judge_model, question=case["question"]
        ),
        relevance=lambda question, value: check_relevance(
            question, value, model=judge_model
        ),
        response_assessment=assessment,
    )
    return asdict(score)


def run_case(
    case: dict,
    judge_model: str,
    skip_judge: bool,
    *,
    correction_mode: str = DEFAULT_CORRECTION_MODE,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[dict, object | None]:
    graph = build_graph(
        correction_mode=correction_mode,
        max_retries=0 if correction_mode == "off" else max_retries,
        review_model=judge_model,
    )
    thread_id = f"comparison-eval-{case['id'].lower()}-{uuid.uuid4().hex[:8]}"
    _seed_setup(graph, thread_id, case["setup"], _scope_catalog())
    started = time.monotonic()
    final = run_turn(
        graph,
        thread_id,
        case["question"],
        include_guard_diagnostics=True,
    )
    wall_ms = round((time.monotonic() - started) * 1000, 2)
    # If a final guard blocks the draft, score retrieval/orchestration from the
    # safe pre-rollback metadata while retaining the guard failure as outcome.
    measured = dict(final.get("guard_diagnostics") or final)
    if final.get("guard_reason"):
        measured["guard_reason"] = final["guard_reason"]
    mechanical = score_comparison_execution(case, measured)
    generation_error = None
    try:
        generation = None if skip_judge else _generation_score(case, final, judge_model)
    except Exception as exc:
        # Judge failures are missing measurements, not zero-quality answers and
        # not reasons to discard otherwise valid retrieval/orchestration data.
        generation = None
        generation_error = f"judge_error:{type(exc).__name__}"
    final_status = final.get("final_status")
    record = {
        "case_id": case["id"],
        "family": case["family"],
        "split": "dev" if case["id"].startswith("C-D") else "holdout",
        "thread_id": thread_id,
        "question": case["question"],
        "resolved_question": final.get("resolved_question"),
        "status": (
            "clarification" if final.get("__interrupt__")
            else "guard_blocked" if final.get("guard_reason")
            else final_status if final_status in {
                "limited", "review_error", "scope_error"
            }
            else "completed"
        ),
        "final_status": final_status,
        "answer": None if final.get("__interrupt__") else final.get("answer"),
        "clarification": (
            final["__interrupt__"][0].value if final.get("__interrupt__") else None
        ),
        "scopes": measured.get("comparison_scopes") or [],
        "context_ids_by_scope": {
            key: [chunk["chunk_id"] for chunk in chunks]
            for key, chunks in (measured.get("comparison_context") or {}).items()
        },
        "nodes_executed": [event["node"] for event in measured.get("node_events", [])],
        "review_log": final.get("review_log") or [],
        "guard_reason": final.get("guard_reason"),
        "guard_events": final.get("guard_events") or [],
        "mechanical_metrics": mechanical,
        "generation_metrics": generation,
        "generation_error": generation_error,
        "wall_latency_ms": wall_ms,
        "graph_usage": measured.get("usage_summary") or {},
        "request_usage": final.get("request_usage") or {},
        "guard_usage": final.get("guard_usage") or {},
    }
    return record, generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=FILES, required=True)
    parser.add_argument("--case", nargs="*", default=[])
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--correction-mode",
        choices=("off", "post_rerank", "post_generation"),
        default=DEFAULT_CORRECTION_MODE,
        help="Graph architecture to evaluate; defaults to the chat service path.",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    dataset = json.loads((SUITE / FILES[args.split]).read_text())
    selected = [case for case in dataset["cases"] if not args.case or case["id"] in args.case]
    missing = set(args.case) - {case["id"] for case in selected}
    if missing:
        raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")

    records, generation_scores = [], []
    for case in selected:
        record, generation = run_case(
            case,
            args.judge_model,
            args.skip_judge,
            correction_mode=args.correction_mode,
            max_retries=args.max_retries,
        )
        records.append(record)
        if generation is not None:
            from src.eval.generation_metrics import GenerationScore
            generation_scores.append(GenerationScore(**generation))
        print(
            f"{case['id']}: route={record['mechanical_metrics']['behavior_pass']} "
            f"scopes={record['mechanical_metrics']['both_scope_accuracy']} "
            f"bilateral={record['mechanical_metrics']['bilateral_full_coverage']} "
            f"latency_ms={record['wall_latency_ms']}"
        )

    payload = {
        "suite": "comparison_eval_v1",
        "split": args.split,
        "run_id": uuid.uuid4().hex,
        "judge_model": None if args.skip_judge else args.judge_model,
        "judge_contract": (
            None if args.skip_judge else {
                "evaluator_version": EVALUATOR_VERSION,
                "rubric_hashes": rubric_hashes(),
            }
        ),
        "graph_config": {
            "correction_mode": args.correction_mode,
            "max_retries": 0 if args.correction_mode == "off" else args.max_retries,
        },
        "records": records,
        "summary": {
            "mechanical": aggregate_comparison([
                record["mechanical_metrics"] for record in records
            ]),
            "generation": (
                aggregate_generation(generation_scores) if generation_scores else None
            ),
        },
    }
    target = args.out or ROOT / "results" / f"comparison_{args.split}_{payload['run_id'][:10]}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {target}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
