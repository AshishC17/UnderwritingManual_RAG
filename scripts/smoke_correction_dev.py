"""Small opt-in dev smoke run; not a benchmark or a calibrated judge audit.

Only question text enters the application. Reference labels are used afterwards.
Saves fresh output, never overwrites a previous run or modifies any dataset.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_retry_eval import load_split, source_ref_map, grade_evidence, grade_answer, build_metrics, prompt_hashes
from src.graph.pipeline import build_graph
from src.graph.runtime import run_turn
from src.generate.generator import build_context, MODEL
from src.eval.judge import JUDGE_MODEL, check_relevance, classify_response
from src.util.telemetry import capture, summarize


def grade(case, answer, chunks, scope):
    started = time.perf_counter()
    with capture() as events:
        judgments, grounded = grade_answer(case, answer, build_context(chunks), JUDGE_MODEL)
        response = classify_response(case["question"], answer, model=JUDGE_MODEL)
        relevant = check_relevance(case["question"], answer, model=JUDGE_MODEL)
    evidence = grade_evidence(case, [c["chunk_id"] for c in chunks], source_ref_map())
    metrics = build_metrics(case, "completed", judgments, grounded, evidence, scope, response)
    metrics["answer_relevance"] = int(relevant)
    return {"metrics": metrics, "evidence": evidence, "judgments": judgments,
            "grounded_counts": grounded, "response": response,
            "evaluation_usage": summarize([{"elapsed_ms": (time.perf_counter()-started)*1000, "events": events}])}


def main():
    run_id = "correction-smoke-" + uuid.uuid4().hex[:10]
    output = Path("results") / (run_id + ".json")
    output.parent.mkdir(exist_ok=True)
    report = {"run_id": run_id, "kind": "two-case dev smoke, not calibrated accuracy",
              "generator": MODEL, "reviewer": JUDGE_MODEL, "evaluator": JUDGE_MODEL,
              "rubrics": prompt_hashes(), "cases": [], "planned": 2}
    # Exclusive creation protects all existing results; checkpoints preserve a
    # partial run if quota/provider failures interrupt later work.
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print("OUTPUT " + str(output), flush=True)
    graph = build_graph(1, correction_mode="post_generation")
    cases = {c["id"]: c for c in load_split("dev")["cases"]}
    for case_id in ("R-D04", "R-D07"):
        case = cases[case_id]
        record = {"case_id": case_id, "question": case["question"]}
        report["cases"].append(record)
        try:
            print("START " + case_id, flush=True)
            state = run_turn(graph, run_id + ":" + case_id, case["question"])
            record["state"] = {k: state.get(k) for k in (
                "answer", "scope", "resolved_question", "metadata_filter", "context_chunks", "sub_queries",
                "final_status", "retry_attempt", "review_log", "node_events", "usage_summary", "correction_error")}
            record["interrupted"] = bool(state.get("__interrupt__"))
            output.write_text(json.dumps(report, indent=2))
            print("GRAPH " + case_id + " " + json.dumps({k: state.get(k) for k in ("final_status", "retry_attempt", "usage_summary")}), flush=True)
            if record["interrupted"]:
                record["clarification"] = str(state["__interrupt__"][0].value)
                continue
            if state.get("final_status") in {"review_error", "scope_error"}:
                record["evaluation_skipped"] = (
                    "runtime review/source guard failed; do not spend judge quota grading the safety fallback"
                )
                print("SKIP_GRADE " + case_id + " " + state["final_status"], flush=True)
                continue
            record["final_grade"] = grade(case, state["answer"], state.get("context_chunks", []), state.get("scope"))
            output.write_text(json.dumps(report, indent=2))
            print("GRADE " + case_id + " " + json.dumps(record["final_grade"]["metrics"]), flush=True)
            reviews = [r for r in state.get("review_log", []) if r.get("stage") == "answer_review"]
            if reviews:
                initial = reviews[0]
                initial_ids = [c["chunk_id"] for c in initial["context"]]
                final_ids = [c["chunk_id"] for c in state.get("context_chunks", [])]
                if initial["draft"] == state["answer"] and initial_ids == final_ids:
                    record["initial_grade"] = record["final_grade"]
                    record["comparison"] = "unchanged draft and context; grade reused, no duplicate judge calls"
                else:
                    # Checkpointed logs retain source hashes. Recover the initial
                    # context from the retained pool only if the text still matches.
                    import hashlib
                    pool = state.get("retrieved_candidates", {})
                    initial_chunks = [pool[cid]["chunk"] for cid in initial_ids]
                    if any(hashlib.sha256(c["text"].encode()).hexdigest() != ref["sha256"]
                           for c, ref in zip(initial_chunks, initial["context"])):
                        raise ValueError("initial context changed; cannot make paired comparison")
                    record["initial_grade"] = grade(case, initial["draft"], initial_chunks, state.get("scope"))
                    record["comparison"] = "initial draft versus delivered response in the same run; not an independent randomized baseline"
                    print("INITIAL_GRADE " + case_id + " " + json.dumps(record["initial_grade"]["metrics"]), flush=True)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            print("ERROR " + case_id + " " + type(exc).__name__, flush=True)
        finally:
            output.write_text(json.dumps(report, indent=2))
    print("DONE " + str(output), flush=True)


if __name__ == "__main__":
    main()
