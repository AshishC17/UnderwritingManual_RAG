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
    claim_recall: float
    forbidden_total: int
    forbidden_asserted: list[str] = field(default_factory=list)
    hallucinated: int = 0
    citations_made: int = 0
    citations_valid: int = 0
    citation_validity: float = 1.0
    # Groundedness — is every claim the answer makes supported by the context?
    claims_made: int = 0
    claims_grounded: int = 0
    groundedness: float = 1.0
    ungrounded_claims: list[str] = field(default_factory=list)
    # Abstention — did it decline when evidence was missing, or answer anyway?
    abstained: int = 0
    evidence_complete: int = 1
    unsupported_confidence: int = 0
    over_refusal: int = 0
    # Relevance — does the answer address the question at all?
    addresses_question: int = 1


def evidence_is_complete(case: dict, retrieved_ids: list[str]) -> bool:
    """Did retrieval supply at least one chunk from every evidence group?

    Abstention can only be judged against this: declining when the evidence was
    there is over-refusal, while answering confidently when it was not is the
    failure that produces confidently wrong output.
    """
    top = set(retrieved_ids)
    return all(
        top & set(g.get("any_of_chunk_ids", []))
        for g in case.get("evidence_groups", [])
    ) if case.get("evidence_groups") else True


def score_generation(
    case: dict,
    answer: str,
    retrieved_ids: list[str],
    judge,
    context: str | None = None,
    decompose=None,
    supported=None,
    relevance=None,
) -> GenerationScore:
    """`judge` is a callable (claim, answer) -> (present: bool, evidence: str).

    The optional callables enable the deeper checks; omitting them keeps the
    original three metrics and leaves the rest at their neutral defaults.
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
        for c, ok in zip(claims, supported(claims, context)):
            if ok:
                grounded += 1
            else:
                ungrounded.append(c)

    # --- abstention ---------------------------------------------------
    abstained = int(bool(ABSTENTION_RE.search(answer)))
    complete = int(evidence_is_complete(case, retrieved_ids))
    # Answered confidently on incomplete evidence — how X08 went wrong.
    unsupported_conf = int(not complete and not abstained)
    # Declined despite having what it needed.
    over_refusal = int(complete and abstained)

    addresses = int(relevance(case["question"], answer)) if relevance else 1

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
        claims_made=made,
        claims_grounded=grounded,
        groundedness=(grounded / made) if made else 1.0,
        ungrounded_claims=ungrounded,
        abstained=abstained,
        evidence_complete=complete,
        unsupported_confidence=unsupported_conf,
        over_refusal=over_refusal,
        addresses_question=addresses,
    )


def aggregate_generation(scores: list[GenerationScore]) -> dict:
    n = len(scores) or 1
    incomplete = [s for s in scores if not s.evidence_complete]
    return {
        "cases": len(scores),
        "claim_recall": sum(s.claim_recall for s in scores) / n,
        "hallucination_rate": sum(s.hallucinated for s in scores) / n,
        "citation_validity": sum(s.citation_validity for s in scores) / n,
        "forbidden_assertions": sum(len(s.forbidden_asserted) for s in scores),
        "groundedness": sum(s.groundedness for s in scores) / n,
        "ungrounded_claims": sum(len(s.ungrounded_claims) for s in scores),
        "answer_relevance": sum(s.addresses_question for s in scores) / n,
        # Denominator is cases with incomplete evidence, not all cases: a system
        # that always retrieves well has no opportunity to answer unsupported.
        "cases_missing_evidence": len(incomplete),
        "unsupported_confidence": (
            sum(s.unsupported_confidence for s in incomplete) / len(incomplete)
            if incomplete else 0.0
        ),
        "over_refusal": sum(s.over_refusal for s in scores) / n,
    }


def by_difficulty_generation(scores: list[GenerationScore]) -> dict[str, dict]:
    order = ["easy", "medium", "difficult", "extreme"]
    buckets: dict[str, list[GenerationScore]] = {}
    for s in scores:
        buckets.setdefault(s.difficulty, []).append(s)
    return {d: aggregate_generation(buckets[d]) for d in order if d in buckets}
