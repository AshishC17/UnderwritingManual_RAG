"""Run eval/retry_eval_v1 through the real graph and grade the answers.

    python scripts/run_retry_eval.py --arm no_retry --split dev --judge-model qwen/qwen3.8-27b
    python scripts/run_retry_eval.py --arm no_retry --split dev --case R-D01 R-D02

Arm `no_retry` is Run A: the baseline. One attempt per case, no gate, no retry.
Its purpose is to produce (a) graded answers and (b) an immutable record of the
initial retrieval state, so a later retry arm can be compared against the same
starting point rather than a second, uncontrolled decomposition.

Why a new runner rather than reusing an existing one, per the suite's README:
`run_generation_eval.py` calls search/rerank/generate directly, bypassing the
graph, and passes no scope filter — it cannot address a four-document corpus.
`run_graph_dev.py` invokes the graph but scores retrieval only; it never grades
a generated answer and records no attempt structure.

Leakage boundary, enforced here and not merely documented: the application
receives `question` and nothing else. Expected scope, answerability, reference
answer, required/forbidden claims, aspects and evidence groups are read only
after the graph has returned, and only by the grading code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.eval.judge import (  # noqa: E402
    JUDGE_MODEL,
    check_supported_batch,
    decompose_claims,
    judge_claim,
    classify_response,
    response_type_pass,
    rubric_hashes,
)
from src.generate.generator import MODEL as GEN_MODEL  # noqa: E402
from src.generate.generator import SYSTEM as GEN_SYSTEM  # noqa: E402
from src.generate.generator import build_context  # noqa: E402
from src.graph.pipeline import FINAL_K, build_graph  # noqa: E402
from src.graph.runtime import run_turn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EVAL_DIR = ROOT / "eval" / "retry_eval_v1"
SPLITS = {"dev": "dev_eval.json", "holdout": "holdout_eval.json"}
DIAGNOSABLE = {"dev"}  # holdout is a verdict; same policy as run_eval.py


# ---------------------------------------------------------------- provenance

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def code_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=ROOT, timeout=5,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}" if rev else "unknown"
    except Exception:
        return "unknown"


def prompt_hashes() -> dict:
    """Record all evaluator rubrics; changed claim prompts invalidate their cache.

    Generator cache invalidation is still a separate application concern.
    """
    return {"generator_system": _sha(GEN_SYSTEM), **rubric_hashes()}


# ---------------------------------------------------------------- eval assets

def load_split(split: str) -> dict:
    return json.loads((EVAL_DIR / SPLITS[split]).read_text())


def source_ref_map() -> dict[str, str]:
    return {
        ref: entry["chunk_id"]
        for ref, entry in json.loads((EVAL_DIR / "sources.json").read_text())["sources"].items()
    }


def corpus_sha() -> str:
    snap = json.loads((EVAL_DIR / "sources.json").read_text()).get("corpus_snapshot", {})
    return snap.get("chunks_sha256", "unknown")


# ---------------------------------------------------------------- artifacts

def build_attempt(state: dict, elapsed_ms: float, status: str) -> dict:
    """The immutable record of what retrieval actually did on this attempt.

    `context_chunk_ids` is the final k handed to the generator — not the whole
    reranked pool. A chunk ranked 11th of 24 never reached the model, so
    scoring coverage against the pool would credit evidence the answer could
    not have used.
    """
    reranked = state.get("reranked") or []
    context = state.get("context_chunks") or []
    details = state.get("rerank_details") or []

    scores = []
    for d in details:
        for s in d.get("rerank_scores", []):
            scores.append({
                "chunk_id": d["chunk_id"],
                "query_index": s.get("sub_query_index", 0),
                "score": float(s.get("rerank_score", 0.0)),
            })

    sub_queries = state.get("sub_queries") or []
    scored_indices = {s["query_index"] for s in scores}
    unscored = [i for i in range(len(sub_queries)) if i not in scored_indices]

    action = {"completed": "generate", "clarification": "clarify"}.get(status, "error")

    usage = state.get("usage_summary") or {}
    return {
        "attempt": 0,
        "retry_of": None,
        "queries": sub_queries,
        "metadata_filter": state.get("metadata_filter"),
        "candidate_chunk_ids": [c["chunk_id"] for c, _ in reranked],
        "context_chunk_ids": [c["chunk_id"] for c in context],
        "rerank_scores": scores,
        "assessment": {
            # No gate exists in this arm. Recording "not_assessed" rather than
            # inventing a verdict keeps the baseline honest.
            "status": "not_assessed",
            "reason": "no_retry arm: no gate is evaluated before generation",
            "unscored_query_indices": unscored,
        },
        "actual_action": action,
        "targeted_query_indices": [],
        "change": {
            "kind": "initial",
            "description": "first and only retrieval pass",
            "effective_request_changed": False,
        },
        "oracle": {},  # filled by grade_evidence
        "usage": {
            "elapsed_ms": round(elapsed_ms, 1),
            "provider_calls": usage.get("api_calls"),
            "tokens": usage.get("known_total_tokens") if usage.get("calls_without_token_usage") == 0 else None,
            "cost_usd": usage.get("estimated_api_cost_usd"),
            "cache_hits": usage.get("cache_hits"),
            "transport_retries": None,  # SDK-internal HTTP attempts are not observable
        },
        "trace_id": None,
    }


# ---------------------------------------------------------------- grading

def grade_evidence(case: dict, context_ids: list[str], ref_map: dict[str, str]) -> dict:
    """Authored evidence groups against the FINAL context. Evaluator-side only.

    AND across groups, OR across each group's source alternatives. No groups
    at all (clarification, or a genuinely absent policy) means coverage is
    null — not 100% from a vacuously satisfied AND.
    """
    groups = case.get("evidence_groups") or []
    if not groups:
        return {
            "evidence_status": "not_applicable",
            "covered_groups": [], "missing_groups": [], "lost_groups": [],
            "rationale": "case authors no positive evidence groups",
        }

    top = set(context_ids)
    covered, missing = [], []
    for g in groups:
        ids = {ref_map[r] for r in g["any_of_source_refs"] if r in ref_map}
        (covered if top & ids else missing).append(g["name"])

    return {
        "evidence_status": "sufficient" if not missing else "insufficient",
        "covered_groups": covered,
        "missing_groups": missing,
        "lost_groups": [],
        "rationale": (
            f"{len(covered)}/{len(groups)} authored groups present in the final "
            f"{len(context_ids)} chunks sent to the generator"
        ),
    }


def grade_answer(case: dict, answer: str, context: str, judge_model: str) -> tuple[list, dict]:
    """Revised evaluator: semantic calibration is pending, not inherited from
    the earlier claim-judge audit. Groundedness includes scenario premises.
    """
    judgments = []

    for i, claim in enumerate(case["required_claims"]):
        present, evidence = judge_claim(claim, answer, model=judge_model)
        judgments.append({
            "kind": "required", "claim_index": i,
            "verdict": "present" if present else "absent",
            "answer_quote": (evidence or "")[:300],
            "source_chunk_ids": [], "rationale": "", "reviewer": "judge",
        })

    for i, claim in enumerate(case.get("forbidden_claims") or []):
        present, evidence = judge_claim(claim, answer, model=judge_model)
        judgments.append({
            "kind": "forbidden", "claim_index": i,
            "verdict": "present" if present else "absent",
            "answer_quote": (evidence or "")[:300],
            "source_chunk_ids": [], "rationale": "", "reviewer": "judge",
        })

    grounded = {"made": 0, "supported": 0}
    if context:
        made = decompose_claims(answer, model=judge_model)
        grounded["made"] = len(made)
        for i, (claim, ok) in enumerate(zip(made, check_supported_batch(made, context, model=judge_model, question=case["question"]))):
            grounded["supported"] += int(ok)
            judgments.append({
                "kind": "generated_claim", "claim_index": i,
                "verdict": "supported" if ok else "unsupported",
                "answer_quote": claim[:300],
                "source_chunk_ids": [], "rationale": "", "reviewer": "judge",
            })

    return judgments, grounded


def aspect_coverage(case: dict, judgments: list) -> dict:
    """An aspect is covered only when EVERY claim it references is present.

    Saying something topical about the aspect is not coverage — that is the
    distinction the authored `claim_indices` exist to enforce, and the reason
    a whole-answer relevance check cannot substitute for this.
    """
    present = {
        j["claim_index"] for j in judgments
        if j["kind"] == "required" and j["verdict"] == "present"
    }
    covered = []
    for aspect in case["required_aspects"]:
        if all(i in present for i in aspect["claim_indices"]):
            covered.append(aspect["id"])
    total = len(case["required_aspects"])
    return {
        "covered": covered,
        "total": total,
        "ratio": (len(covered) / total) if total else None,
    }


def build_metrics(case: dict, status: str, judgments: list, grounded: dict,
                  oracle: dict, scope: dict | None, response_assessment: dict | None = None) -> dict:
    required = [j for j in judgments if j["kind"] == "required"]
    forbidden = [j for j in judgments if j["kind"] == "forbidden"]
    n_req = len(required)
    n_present = sum(1 for j in required if j["verdict"] == "present")
    forbidden_hit = any(j["verdict"] == "present" for j in forbidden)

    answered = status == "completed"
    expected = case["expected_response"]
    if status == "clarification":
        expected_pass = expected == "clarify"  # graph route, not wording quality
    elif answered and response_assessment:
        expected_pass = response_type_pass(expected, response_assessment["kind"])
    else:
        expected_pass = False if not answered else None

    exp_scope = case.get("expected_scope")
    if exp_scope is None or scope is None:
        scope_correct = None
    else:
        scope_correct = (
            scope.get("lender_id") == exp_scope["lender_id"]
            and scope.get("document_version") == exp_scope["document_version"]
        )

    return {
        "claim_recall": (n_present / n_req) if (answered and n_req) else None,
        "groundedness": (
            (grounded["supported"] / grounded["made"])
            if (answered and grounded["made"]) else None
        ),
        "full_claim_coverage": (n_present == n_req and not forbidden_hit) if answered and n_req else None,
        "forbidden_claim_hit": forbidden_hit if answered else None,
        "expected_response_pass": expected_pass,
        "context_group_coverage": (
            (len(oracle["covered_groups"]) /
             (len(oracle["covered_groups"]) + len(oracle["missing_groups"])))
            if answered and (oracle.get("covered_groups") or oracle.get("missing_groups")) else None
        ),
        "scope_correct": scope_correct,
        # Deliberately null: whether the answer preserved every scenario
        # constraint is a semantic judgement, and no validated check for it
        # exists. Nulling is honest; a keyword match would look like a measure
        # without being one.
        "scenario_preserved": None,
        "pipeline_success": status in ("completed", "clarification"),
    }


# ---------------------------------------------------------------- main

def run_case(app, case: dict, run_id: str, ref_map: dict[str, str],
             judge_model: str, config: dict) -> dict:
    thread_id = f"{run_id}:{case['id']}"
    started = time.monotonic()

    try:
        # Only the question crosses this line.
        final = run_turn(app, thread_id, case["question"])
        interrupted = bool(final.get("__interrupt__"))
        status = "clarification" if interrupted else "completed"
        error = None
    except Exception as exc:
        final, status, error = {}, "error", f"{type(exc).__name__}: {exc}"

    elapsed_ms = (time.monotonic() - started) * 1000
    attempt = build_attempt(final, elapsed_ms, status)

    answer = None if status != "completed" else (final.get("answer") or "")
    context_chunks = final.get("context_chunks") or []
    oracle = grade_evidence(case, attempt["context_chunk_ids"], ref_map)
    if status != "completed":
        oracle = {
            "evidence_status": "not_applicable", "covered_groups": [],
            "missing_groups": [], "lost_groups": [],
            "rationale": f"Evidence coverage unassessed: run stopped with status {status}; score response/routing separately.",
        }
    attempt["oracle"] = oracle

    judgments, grounded = [], {"made": 0, "supported": 0}
    if status == "completed" and answer:
        judgments, grounded = grade_answer(
            case, answer, build_context(context_chunks), judge_model
        )

    response_assessment = (
        classify_response(case["question"], answer or "", model=judge_model)
        if status == "completed" else None
    )
    metrics = build_metrics(case, status, judgments, grounded, oracle, final.get("scope"), response_assessment)

    record = {
        "run_id": run_id,
        "case_id": case["id"],
        "arm": "no_retry",
        "status": status,
        "config": config,
        "attempts": [attempt],
        "final_answer": answer,
        "final_context_chunk_ids": attempt["context_chunk_ids"],
        "claim_judgments": judgments,
        "metrics": metrics,
        "response_assessment": response_assessment,
    }
    # Not in the record schema; kept alongside for reporting.
    record["_aspects"] = aspect_coverage(case, judgments) if judgments else None
    record["_error"] = error
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["no_retry"], default="no_retry")
    ap.add_argument("--split", choices=list(SPLITS), default="dev")
    ap.add_argument("--case", nargs="+", default=None)
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

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    config = {
        "corpus_sha256": corpus_sha(),
        "code_revision": code_revision(),
        "config_sha256": _sha(f"{GEN_MODEL}|{args.judge_model}|{FINAL_K}|no_retry"),
        "generator_model": GEN_MODEL,
        "judge_model": args.judge_model,
        "prompt_hashes": prompt_hashes(),
        "max_retries": 0,
        "final_k": FINAL_K,
        "score_threshold": None,
        "cache_mode": "warm",
    }

    print(f"suite: retry_eval_v1   arm: {args.arm}   split: {args.split}   cases: {len(cases)}")
    print(f"run_id: {run_id}   generator: {GEN_MODEL}   judge: {args.judge_model}")
    if protected:
        print("HOLDOUT MODE — aggregates only; per-case detail suppressed by policy.")
    print()

    app = build_graph()
    records, partial = [], False
    for i, case in enumerate(cases, 1):
        t0 = time.monotonic()
        try:
            rec = run_case(app, case, run_id, source_ref_map(), args.judge_model, config)
        except Exception as exc:
            print(f"\n  STOPPED at {case['id']}: {type(exc).__name__}: {str(exc)[:160]}")
            print(f"  reporting the {len(records)} case(s) that completed.")
            partial = True
            break
        records.append(rec)
        m = rec["metrics"]
        if protected:
            print(f"  [{i}/{len(cases)}] scored", flush=True)
        else:
            cr = "  n/a" if m["claim_recall"] is None else f"{m['claim_recall']:5.2f}"
            gr = "  n/a" if m["groundedness"] is None else f"{m['groundedness']:5.2f}"
            cov = "  n/a" if m["context_group_coverage"] is None else f"{m['context_group_coverage']:5.2f}"
            print(f"  [{i}/{len(cases)}] {case['id']:<7} {rec['status']:<13} "
                  f"claims {cr}  grounded {gr}  evidence {cov}  {time.monotonic()-t0:5.1f}s")

    if not records:
        raise SystemExit("no case completed")

    # ---- report --------------------------------------------------------
    def ratio(key):
        vals = [r["metrics"][key] for r in records if r["metrics"][key] is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def count(key):
        vals = [r["metrics"][key] for r in records if r["metrics"][key] is not None]
        return (sum(1 for v in vals if v), len(vals))

    print("\n" + "=" * 74)
    print(f"RUN A (no_retry) — {args.split}: {len(records)} of {len(cases)} cases")
    print("=" * 74)
    from collections import Counter
    print("  status:", dict(Counter(r["status"] for r in records)))
    for key in ("claim_recall", "groundedness", "context_group_coverage"):
        val, n = ratio(key)
        print(f"  {key:<24} {'n/a' if val is None else f'{val:.3f}'}   (n={n})")
    for key in ("full_claim_coverage", "forbidden_claim_hit", "expected_response_pass", "scope_correct"):
        hit, n = count(key)
        print(f"  {key:<24} {hit}/{n}")
    asp = [r["_aspects"]["ratio"] for r in records if r.get("_aspects") and r["_aspects"]["ratio"] is not None]
    if asp:
        print(f"  {'aspect_coverage':<24} {sum(asp)/len(asp):.3f}   (n={len(asp)})")
    print("  scenario_preserved       null — no validated check; not asserted")

    if not protected:
        print("\nper-case detail")
        for r in records:
            oracle = r["attempts"][0]["oracle"]
            a = r.get("_aspects")
            aspects = "n/a" if not a else f"{len(a['covered'])}/{a['total']}"
            missing = ",".join(oracle["missing_groups"]) or "-"
            print(f"  {r['case_id']:<7} {r['status']:<13} "
                  f"evidence={oracle['evidence_status']:<12} "
                  f"aspects={aspects:<6} missing_groups={missing}")

    out = Path(args.out) if args.out else ROOT / "results" / f"retry_{args.arm}_{args.split}.json"
    if partial:
        out = out.with_suffix(".partial.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "suite": "retry_eval_v1", "arm": args.arm, "split": args.split,
        "run_id": run_id, "config": config, "records": records,
        "planned_cases": len(cases), "recorded_cases": len(records),
        "metric_denominators": {key: ratio(key)[1] for key in records[0]["metrics"]},
    }, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
