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
    )


def aggregate(scores: list[CaseScore]) -> dict:
    n = len(scores) or 1
    return {
        "cases": len(scores),
        "group_recall": sum(s.group_recall for s in scores) / n,
        "full_coverage": sum(s.full_coverage for s in scores) / n,
        "precision": sum(s.precision for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
    }


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
