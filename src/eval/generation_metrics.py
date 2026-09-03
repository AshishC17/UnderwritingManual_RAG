"""Generation metrics scored from required_claims and forbidden_claims.

Three published metrics, no invention:

- **claim recall** — fraction of `required_claims` the answer asserts. The
  RAGAS/TruLens family calls the same idea answer completeness.
- **hallucination rate** — fraction of questions where the answer asserts ANY
  `forbidden_claim`. Binary per question, because one wrong assertion makes an
  underwriting answer unusable regardless of what else was right. This is the
  complement of RAGAS `faithfulness` / TruLens `groundedness`, sharpened by
  pre-specified wrong answers.
- **citation validity** — fraction of cited chunk_ids that were actually in the
  retrieved context, catching invented citations.

`forbidden_claims` here are not generic distractors: they are the adjacent matrix
cell, the over-generalized exception, the confused sibling rule. A model can
retrieve perfectly and still assert one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Models cite with whichever bracket they favour — gpt-oss-120b emits CJK 【】 rather
# than the ASCII [] the prompt asks for. Matching only [] scored every citation
# invalid, so accept both rather than penalise formatting for a substantive metric.
CITATION_RE = re.compile(r"[\[【]([^\]】\s]+::\d+)[\]】]")


@dataclass
class GenerationScore:
    case_id: str
    difficulty: str
    stress_type: str
    claims_required: int
    claims_present: int
    claim_recall: float
    forbidden_total: int
    forbidden_asserted: list[str] = field(default_factory=list)
    hallucinated: int = 0
    citations_made: int = 0
    citations_valid: int = 0
    citation_validity: float = 1.0


def score_generation(
    case: dict,
    answer: str,
    retrieved_ids: list[str],
    judge,
) -> GenerationScore:
    """`judge` is a callable (claim, answer) -> (present: bool, evidence: str)."""
    required = case.get("required_claims", [])
    forbidden = case.get("forbidden_claims", [])

    present = sum(1 for c in required if judge(str(c), answer)[0])
    asserted = [str(c) for c in forbidden if judge(str(c), answer)[0]]

    cited = CITATION_RE.findall(answer)
    valid = [c for c in cited if c in set(retrieved_ids)]

    return GenerationScore(
        case_id=case["id"],
        difficulty=case.get("difficulty", "?"),
        stress_type=case.get("stress_type", "?"),
        claims_required=len(required),
        claims_present=present,
        claim_recall=(present / len(required)) if required else 1.0,
        forbidden_total=len(forbidden),
        forbidden_asserted=asserted,
        hallucinated=int(bool(asserted)),
        citations_made=len(cited),
        citations_valid=len(valid),
        citation_validity=(len(valid) / len(cited)) if cited else 1.0,
    )


def aggregate_generation(scores: list[GenerationScore]) -> dict:
    n = len(scores) or 1
    return {
        "cases": len(scores),
        "claim_recall": sum(s.claim_recall for s in scores) / n,
        "hallucination_rate": sum(s.hallucinated for s in scores) / n,
        "citation_validity": sum(s.citation_validity for s in scores) / n,
        "forbidden_assertions": sum(len(s.forbidden_asserted) for s in scores),
    }


def by_difficulty_generation(scores: list[GenerationScore]) -> dict[str, dict]:
    order = ["easy", "medium", "difficult", "extreme"]
    buckets: dict[str, list[GenerationScore]] = {}
    for s in scores:
        buckets.setdefault(s.difficulty, []).append(s)
    return {d: aggregate_generation(buckets[d]) for d in order if d in buckets}
