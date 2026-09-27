"""Generation metrics scored from required_claims and forbidden_claims.

Project metrics (not interchangeable with similarly named framework metrics):

- **claim recall** — fraction of `required_claims` the answer asserts. The
  RAGAS/TruLens family calls the same idea answer completeness.
- **hallucination rate** — fraction of questions where the answer asserts ANY
  `forbidden_claim`. Binary per question, because one wrong assertion makes an
  underwriting answer unusable regardless of what else was right. This is the
  authored-trap rate, NOT the complement of groundedness and NOT an exhaustive
  hallucination detector. The historical field name is kept for compatibility.
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


# Phrases a model uses when it declines to answer from the given context. Used to
# separate a principled "the context does not say" from a confident invention.
ABSTENTION_RE = re.compile(
    r"\b(?:do(?:es)? not (?:contain|specify|state|provide|include)"
    r"|not (?:stated|specified|provided|available|present) in the (?:context|excerpts?)"
    r"|no information (?:about|on|regarding)"
    r"|cannot (?:be )?(?:determine|answer|establish)"
    r"|insufficient (?:context|information)"
    r"|i (?:don'?t|do not) know)\b",
    re.I,
)


@dataclass
class GenerationScore:
    case_id: str
    difficulty: str
    stress_type: str
    claims_required: int
    claims_present: int
    claim_recall: float | None
    forbidden_total: int
    forbidden_asserted: list[str] = field(default_factory=list)
    hallucinated: int = 0
    citations_made: int = 0
    citations_valid: int = 0
    citation_validity: float | None = None
    # Groundedness — is every claim the answer makes supported by the context?
    claims_made: int = 0
    claims_grounded: int = 0
    groundedness: float | None = None
    ungrounded_claims: list[str] = field(default_factory=list)
    # Abstention — did it decline when evidence was missing, or answer anyway?
    abstained: int = 0
    evidence_complete: int | None = None
    unsupported_confidence: int | None = None
    over_refusal: int | None = None
    response_kind: str | None = None
    # Relevance — does the answer address the question at all?
    addresses_question: int | None = None


def evidence_is_complete(case: dict, retrieved_ids: list[str]) -> bool | None:
    """Did retrieval supply at least one chunk from every evidence group?

    Abstention can only be judged against this: declining when the evidence was
    there is over-refusal, while answering confidently when it was not is the
    failure that produces confidently wrong output.
    """
    top = set(retrieved_ids)
    return all(
        top & set(g.get("any_of_chunk_ids", []))
        for g in case.get("evidence_groups", [])
    ) if case.get("evidence_groups") else None


def score_generation(
    case: dict,
    answer: str,
    retrieved_ids: list[str],
    judge,
    context: str | None = None,
    decompose=None,
    supported=None,
    relevance=None,
    response_assessment: dict | None = None,
) -> GenerationScore:
    """`judge` is a callable (claim, answer) -> (present: bool, evidence: str).

    The optional callables enable the deeper checks; omitting them keeps the
    original three metrics; unmeasured groundedness remains null, not a pass.
    """
    required = case.get("required_claims", [])
    forbidden = case.get("forbidden_claims", [])

    present = sum(1 for c in required if judge(str(c), answer)[0])
    asserted = [str(c) for c in forbidden if judge(str(c), answer)[0]]

    cited = CITATION_RE.findall(answer)
    valid = [c for c in cited if c in set(retrieved_ids)]

    # --- groundedness -------------------------------------------------
    # `supported` takes the whole claim list at once: sending the context with
    # every individual claim multiplied token spend by the number of claims.
    made, grounded, ungrounded = 0, 0, []
    if decompose and supported and context is not None:
        claims = decompose(answer)
        made = len(claims)
        verdicts = supported(claims, context)
        if len(verdicts) != len(claims) or any(type(v) is not bool for v in verdicts):
            raise ValueError("groundedness requires one boolean verdict per claim")
        for c, ok in zip(claims, verdicts):
            if ok:
                grounded += 1
            else:
                ungrounded.append(c)

    # --- abstention ---------------------------------------------------
    response_kind = response_assessment.get("kind") if response_assessment else None
    abstained = int(response_kind in {"abstention", "clarification", "qualified_answer"}) if response_kind else int(bool(ABSTENTION_RE.search(answer)))
    complete_value = evidence_is_complete(case, retrieved_ids)
    complete = int(complete_value) if complete_value is not None else None
    # Answered confidently on incomplete evidence — how X08 went wrong.
    unsupported_conf = int(not complete and not abstained) if complete is not None else None
    # Declined despite having what it needed.
    over_refusal = int(complete and (response_kind == "abstention" if response_kind else abstained)) if complete is not None else None

    addresses = int(relevance(case["question"], answer)) if relevance else None

    return GenerationScore(
        case_id=case["id"],
        difficulty=case.get("difficulty", "?"),
        stress_type=case.get("stress_type", "?"),
        claims_required=len(required),
        claims_present=present,
        claim_recall=(present / len(required)) if required else None,
        forbidden_total=len(forbidden),
        forbidden_asserted=asserted,
        hallucinated=int(bool(asserted)),
        citations_made=len(cited),
        citations_valid=len(valid),
        citation_validity=(len(valid) / len(cited)) if cited else None,
        claims_made=made,
        claims_grounded=grounded,
        groundedness=(grounded / made) if made else None,
        ungrounded_claims=ungrounded,
        abstained=abstained,
        evidence_complete=complete,
        unsupported_confidence=unsupported_conf,
        over_refusal=over_refusal,
        response_kind=response_kind,
        addresses_question=addresses,
    )


def aggregate_generation(scores: list[GenerationScore]) -> dict:
    n = len(scores) or 1
    incomplete = [s for s in scores if s.evidence_complete == 0]
    def mean(field):
        values = [getattr(s, field) for s in scores if getattr(s, field) is not None]
        return sum(values) / len(values) if values else None
    return {
        "cases": len(scores),
        "claim_recall": mean("claim_recall"),
        "hallucination_rate": sum(s.hallucinated for s in scores) / n,
        "citation_validity": mean("citation_validity"),
        "forbidden_assertions": sum(len(s.forbidden_asserted) for s in scores),
        "groundedness": mean("groundedness"),
        "ungrounded_claims": sum(len(s.ungrounded_claims) for s in scores),
        "answer_relevance": mean("addresses_question"),
        # Denominator is cases with incomplete evidence, not all cases: a system
        # that always retrieves well has no opportunity to answer unsupported.
        "cases_missing_evidence": len(incomplete),
        "unsupported_confidence": (
            sum(s.unsupported_confidence for s in incomplete) / len(incomplete)
            if incomplete else None
        ),
        "over_refusal": mean("over_refusal"),
        "metric_denominators": {
            field: sum(getattr(s, field) is not None for s in scores)
            for field in ("claim_recall", "groundedness", "citation_validity", "over_refusal", "addresses_question")
        },
    }


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def by_difficulty_generation(scores: list[GenerationScore]) -> dict[str, dict]:
    order = ["easy", "medium", "difficult", "extreme"]
    buckets: dict[str, list[GenerationScore]] = {}
    for s in scores:
        buckets.setdefault(s.difficulty, []).append(s)
    return {d: aggregate_generation(buckets[d]) for d in order if d in buckets}
