"""Grade a saved correction smoke run without rerunning retrieval/generation.

Every semantic metric fails independently. A malformed judge response becomes
N/A plus an evaluator error; it never becomes a false verdict or aborts the
remaining metrics. Writes a new file and preserves the source run.
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
from scripts.run_retry_eval import load_split, source_ref_map, grade_evidence
from src.eval.generation_metrics import CITATION_RE
from src.eval.judge import (JUDGE_MODEL, check_relevance, check_supported_batch,
                            classify_response, decompose_claims, judge_claim,
                            response_type_pass, rubric_hashes)
from src.generate.generator import build_context
from src.util.telemetry import capture, summarize

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def safe(errors: list[dict], stage: str, function):
    try:
        return function()
    except Exception as exc:
        errors.append({"stage": stage, "error_type": type(exc).__name__,
                       "reason": str(exc)[:180]})
        return None


def grade_case(case: dict, state: dict) -> dict:
    answer = state["answer"]
    chunks = state.get("context_chunks", [])
    context = build_context(chunks)
    ids = [c["chunk_id"] for c in chunks]
    errors: list[dict] = []
    judgments = []
    started = time.perf_counter()
    with capture() as events:
        for kind, field in (("required", "required_claims"), ("forbidden", "forbidden_claims")):
            for index, claim in enumerate(case.get(field, [])):
                result = safe(errors, f"{kind}_claim_{index}",
                              lambda claim=claim: judge_claim(claim, answer, model=JUDGE_MODEL))
                judgments.append({"kind": kind, "claim_index": index, "claim": claim,
                                  "verdict": None if result is None else ("present" if result[0] else "absent"),
                                  "answer_quote": "" if result is None else result[1]})

        claims = safe(errors, "decompose_generated_claims",
                      lambda: decompose_claims(answer, model=JUDGE_MODEL))
        support = (safe(errors, "groundedness_support",
                        lambda: check_supported_batch(claims, context, model=JUDGE_MODEL,
                                                      question=case["question"]))
                   if claims is not None else None)
        response = safe(errors, "response_kind",
                        lambda: classify_response(case["question"], answer, model=JUDGE_MODEL))
        relevant = safe(errors, "answer_relevance",
                        lambda: check_relevance(case["question"], answer, model=JUDGE_MODEL))

    required = [j for j in judgments if j["kind"] == "required"]
    forbidden = [j for j in judgments if j["kind"] == "forbidden"]
    req_complete = all(j["verdict"] is not None for j in required)
    forbidden_complete = all(j["verdict"] is not None for j in forbidden)
    present = sum(j["verdict"] == "present" for j in required)
    forbidden_hit = (any(j["verdict"] == "present" for j in forbidden)
                     if forbidden_complete else None)
    evidence = grade_evidence(case, ids, source_ref_map())
    cited = CITATION_RE.findall(answer)
    valid_cited = [cid for cid in cited if cid in set(ids)]
    groundedness = (sum(support) / len(support)
                    if claims is not None and support is not None and claims else None)
    claim_recall = present / len(required) if req_complete and required else None
    response_pass = (response_type_pass(case["expected_response"], response["kind"])
                     if response is not None else None)
    metrics = {
        "evidence_group_coverage": (
            len(evidence["covered_groups"]) /
            (len(evidence["covered_groups"]) + len(evidence["missing_groups"]))
            if evidence["covered_groups"] or evidence["missing_groups"] else None),
        "full_evidence_coverage": not evidence["missing_groups"] if evidence["evidence_status"] != "not_applicable" else None,
        "claim_recall": claim_recall,
        "full_claim_coverage": (claim_recall == 1 and not forbidden_hit)
                               if claim_recall is not None and forbidden_hit is not None else None,
        "forbidden_claim_hit": forbidden_hit,
        "groundedness": groundedness,
        "answer_relevance": int(relevant) if relevant is not None else None,
        "expected_response_pass": response_pass,
        "citation_validity": len(valid_cited) / len(cited) if cited else None,
        "recognized_citations": len(cited),
        "valid_recognized_citations": len(valid_cited),
        "scope_correct": (state.get("scope", {}).get("lender_id") == case["expected_scope"]["lender_id"]
                          and state.get("scope", {}).get("document_version") == case["expected_scope"]["document_version"]),
    }
    elapsed = (time.perf_counter() - started) * 1000
    return {"metrics": metrics, "evidence": evidence, "judgments": judgments,
            "generated_claims": claims, "support_verdicts": support,
            "response_assessment": response, "evaluator_errors": errors,
            "evaluation_usage": summarize([{"elapsed_ms": elapsed, "events": events}])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    cases = {c["id"]: c for c in load_split("dev")["cases"]}
    output = args.source.with_name(args.source.stem + "-graded-" + uuid.uuid4().hex[:8] + ".json")
    report = {"source_run": str(args.source), "judge_model": JUDGE_MODEL,
              "rubric_hashes": rubric_hashes(), "cases": []}
    output.write_text(json.dumps(report, indent=2))
    print("OUTPUT " + str(output), flush=True)
    for saved in source["cases"]:
        case_id = saved["case_id"]
        print("GRADE_START " + case_id, flush=True)
        result = grade_case(cases[case_id], saved["state"])
        report["cases"].append({"case_id": case_id, "result": result})
        output.write_text(json.dumps(report, indent=2))
        print("GRADE_RESULT " + case_id + " " + json.dumps(result["metrics"]), flush=True)
        if result["evaluator_errors"]:
            print("EVALUATOR_ERRORS " + case_id + " " + json.dumps(result["evaluator_errors"]), flush=True)
    print("DONE " + str(output), flush=True)


if __name__ == "__main__":
    main()
