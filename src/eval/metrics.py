"""Retrieval metrics scored against `evidence_groups`.

`evidence_groups` is an AND across groups, OR within each: a case is fully
answerable only when at least one chunk from *every* group was retrieved. That
structure expresses joint conditions plain chunk recall cannot — "the code
definition AND the rule row that uses it" is two groups, and satisfying only one
produces a confidently half-right answer.

Scored against `evidence_groups`, never `supporting_chunk_ids`. The latter is
related enrichment and is frequently disjoint from the evidence actually needed:
for the Table 7 matrix-cell case it points at the surrounding prose and
footnotes, not the matrix holding the answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseScore:
    case_id: str
    difficulty: str
    stress_type: str
    group_recall: float       # fraction of evidence groups satisfied
    full_coverage: int        # 1 only if every group satisfied
    precision: float
    reciprocal_rank: float
    groups_total: int
    groups_hit: int
    missed_groups: list[str]
    # Stage disambiguation — None when the case does not test it.
    stage_correct: int | None = None
    stage_rank: int | None = None       # best-ranked correct-stage chunk
    distractor_rank: int | None = None  # best-ranked wrong-stage chunk


def score_stage(case: dict, retrieved: list[str], k: int):
    """Did the right *stage* win, or did a wrong-stage chunk outrank it?

    Answers the reused-identifier trap the corpus was built around: a question
    about the stage before full underwriting can be answered by a chunk covering
    the same control at the wrong stage, which plain recall scores as a hit.
    Ranks are returned alongside the verdict — a correct chunk ranked 9th with a
    distractor 8th fails differently from one that never appeared at all.
    """
    spec = (case.get("evaluation_metrics") or {}).get("stage_accuracy_at_k") or {}
    if not spec.get("applicable"):
        return None, None, None

    top = retrieved[:k]
    correct = set(spec.get("correct_stage_chunk_ids", []))
    distractors = set(spec.get("distractor_stage_chunk_ids", []))

    c_rank = next((i for i, cid in enumerate(top, 1) if cid in correct), None)
    d_rank = next((i for i, cid in enumerate(top, 1) if cid in distractors), None)

    if c_rank is None:
        return 0, None, d_rank
    if d_rank is None:
        return 1, c_rank, None
    return int(c_rank < d_rank), c_rank, d_rank


def relevant_ids(case: dict) -> set[str]:
    """Union of every chunk that could satisfy any group."""
    out: set[str] = set()
    for g in case.get("evidence_groups", []):
        out.update(g.get("any_of_chunk_ids", []))
    return out


def score_case(case: dict, retrieved: list[str], k: int) -> CaseScore:
    top = retrieved[:k]
    top_set = set(top)
    groups = case.get("evidence_groups", [])

    hit, missed = 0, []
    for g in groups:
        if top_set & set(g.get("any_of_chunk_ids", [])):
            hit += 1
        else:
            missed.append(g.get("name", "?"))

    relevant = relevant_ids(case)
    rr = 0.0
    for i, cid in enumerate(top, start=1):
        if cid in relevant:
            rr = 1.0 / i
            break

    n = len(groups)
    stage_ok, c_rank, d_rank = score_stage(case, retrieved, k)
    return CaseScore(
        case_id=case["id"],
        difficulty=case.get("difficulty", "?"),
        stress_type=case.get("stress_type", "?"),
        group_recall=(hit / n) if n else 0.0,
        full_coverage=int(n > 0 and hit == n),
        # Low by construction: 1-3 relevant chunks out of 87 caps precision@k
        # at roughly 3/k. Logged for completeness, not a headline number.
        precision=(len(top_set & relevant) / k) if k else 0.0,
        reciprocal_rank=rr,
        groups_total=n,
        groups_hit=hit,
        missed_groups=missed,
        stage_correct=stage_ok,
        stage_rank=c_rank,
        distractor_rank=d_rank,
    )


def aggregate(scores: list[CaseScore]) -> dict:
    n = len(scores) or 1
    staged = [s for s in scores if s.stage_correct is not None]
    out = {
        "cases": len(scores),
        "group_recall": sum(s.group_recall for s in scores) / n,
        "full_coverage": sum(s.full_coverage for s in scores) / n,
        "precision": sum(s.precision for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
    }
    if staged:
        out["stage_accuracy"] = sum(s.stage_correct for s in staged) / len(staged)
        out["stage_cases"] = len(staged)
    return out


def by_difficulty(scores: list[CaseScore]) -> dict[str, dict]:
    order = ["easy", "medium", "difficult", "extreme"]
    buckets: dict[str, list[CaseScore]] = {}
    for s in scores:
        buckets.setdefault(s.difficulty, []).append(s)
    return {d: aggregate(buckets[d]) for d in order if d in buckets}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((p / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]
