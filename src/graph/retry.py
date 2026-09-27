"""Corrective-retrieval policy: the gate, and the recovery actions it can choose.

Kept out of `pipeline.py` so the decision logic is a set of pure functions that
can be tested against `eval/retry_eval_v1/gate_fixtures.json` without a graph,
a model, or a network call.

**The gate can only find faults. It can never certify sufficiency.** That is not
conservatism, it is what the fixtures prove: G01 and G02 hold identical
sufficient evidence and score 0.95 and 0.29, while G03 scores 0.95 with the
required exception missing. No function of the rerank score separates them —
a threshold sweep over the full 0-1 range misclassifies at best 5 of 12. So a
clean deterministic pass returns `unknown`, never `sufficient`, and the real
sufficiency question is deferred to a check that can see the generated answer.

Everything here runs on runtime state only. Authored evidence groups, required
claims and aspects are evaluator-side; using them to decide a retry would be
measuring the system against an oracle it will not have in production.
"""

from __future__ import annotations

from typing import Literal

# Cheapest first. Reselection touches no network; a deeper search costs
# retrieval; a reformulation costs an LLM call on top of that. Escalating in
# this order means the expensive actions only run when the cheap ones did not
# recover the answer.
ACTION_LADDER = ("reselect_context", "deeper_search", "reformulate")

DEPTH_MULTIPLIER = 2  # what "deeper" means for a deeper_search attempt


def assess_deterministic(
    context_chunks: list[dict],
    scope: dict | None,
    prior_context_ids: list[str] | None = None,
) -> dict:
    """Fault checks over runtime state. Returns a schema-shaped assessment.

    Three checks, in order of severity:

    1. Empty context — nothing to answer from.
    2. Scope mismatch — a chunk in the final context belongs to a different
       lender or document version than the resolved scope. This is the one
       failure that must never be "fixed" by searching harder: it means the
       filter did not hold, and answering would blend manuals.
    3. Evidence lost — chunks the previous attempt had in its final context
       that this attempt dropped. Detected by chunk-id set difference, not by
       authored evidence groups, so it stays leak-free.
    """
    ids = [c["chunk_id"] for c in context_chunks]

    if not context_chunks:
        return {
            "status": "insufficient",
            "reason": "empty context: retrieval returned nothing to answer from",
            "unscored_query_indices": [],
        }

    if scope and scope.get("status") == "resolved":
        stray = [
            c["chunk_id"] for c in context_chunks
            if c.get("lender_id") != scope.get("lender_id")
            or c.get("document_version") != scope.get("document_version")
        ]
        if stray:
            return {
                "status": "wrong_scope",
                "reason": (
                    f"{len(stray)} of {len(ids)} context chunks fall outside the resolved "
                    f"scope {scope.get('lender_id')}/{scope.get('document_version')}: "
                    + ", ".join(stray[:3])
                ),
                "unscored_query_indices": [],
            }

    # Evidence loss is deliberately NOT a trigger here. A prior chunk leaving
    # the final k is usually a better chunk displacing it — that is what
    # successful recovery looks like, and fixture G08 is exactly that case:
    # the missing exception arrives, filler drops out, and a naive
    # set-difference check calls the recovery a failure. Separating benign
    # displacement from real loss requires knowing which chunks mattered,
    # which is authored ground truth and not available at runtime. So loss is
    # handled two other ways: `check_merge_invariant` guards the mechanism,
    # and group-level gain/loss is reported evaluator-side where the authored
    # groups legitimately exist.

    return {
        "status": "unknown",
        "reason": (
            "no deterministic fault; sufficiency is not decidable from scores or "
            "metadata and is deferred to the post-generation check"
        ),
        "unscored_query_indices": [],
    }


def check_merge_invariant(prior_pool_ids: list[str], merged_pool_ids: list[str]) -> list[str]:
    """Did the retry keep everything the previous attempt had found?

    A guard on the mechanism, not a judgement about sufficiency. `merge_pools`
    is supposed to make this impossible; if it ever returns a non-empty list,
    the retry replaced its predecessor's evidence instead of adding to it —
    the failure mode G10 is built around. Checked against the candidate POOL,
    not the final k, because dropping out of the final k on score is ranking,
    not loss.
    """
    merged = set(merged_pool_ids)
    return [cid for cid in prior_pool_ids if cid not in merged]


def action_for(assessment: dict) -> str:
    """Which recovery a deterministic fault calls for.

    A scope violation is repaired, never retried — searching deeper in the
    wrong manual finds more of the wrong manual.
    """
    if assessment["status"] == "wrong_scope":
        return "repair_scope"
    if assessment["status"] == "insufficient":
        return "retry"
    return "generate"


def assess_answer(
    answer: str,
    sub_queries: list[str],
    ungrounded_claims: list[str],
    unaddressed_queries: list[str],
) -> dict:
    """Post-generation verdict, from checks that need no ground truth.

    Two independent failure modes, deliberately not merged into one score:
    an answer can be fully grounded and still not answer half the question,
    and it can address every part of the question while asserting something
    the context never said.
    """
    problems = []
    if ungrounded_claims:
        problems.append(f"{len(ungrounded_claims)} unsupported claim(s)")
    if unaddressed_queries:
        problems.append(f"{len(unaddressed_queries)} unaddressed part(s) of the question")

    if not problems:
        return {
            "status": "sufficient",
            "reason": "every asserted claim is supported and every part of the question is addressed",
            "unscored_query_indices": [],
        }

    return {
        "status": "insufficient",
        "reason": "; ".join(problems),
        "unscored_query_indices": [
            i for i, q in enumerate(sub_queries) if q in set(unaddressed_queries)
        ],
    }


def recovery_seed(ungrounded_claims: list[str], unaddressed_queries: list[str]) -> str | None:
    """What the next retrieval should go looking for.

    An unaddressed part of the question is the more direct signal — it names an
    information need that went unmet. An unsupported claim is second best: it
    describes an assertion the context could not back, which is a usable
    description of the evidence that is missing.
    """
    if unaddressed_queries:
        return unaddressed_queries[0]
    if ungrounded_claims:
        return ungrounded_claims[0]
    return None


def next_action(attempt: int, pool_size: int, final_k: int, tried: list[str]) -> str | None:
    """Pick the cheapest recovery not yet tried that is actually available.

    Reselection is only meaningful when the pool holds chunks the generator
    never saw; offering it on an exhausted pool would burn an attempt doing
    nothing, which the fixtures count as a wasted retry rather than a recovery.
    """
    for action in ACTION_LADDER:
        if action in tried:
            continue
        if action == "reselect_context" and pool_size <= final_k:
            continue
        return action
    return None


def merge_pools(prior: list[dict], fresh: list[dict]) -> list[dict]:
    """Union by chunk_id, prior evidence first.

    Retry adds to what was found; it does not replace it. Replacement is how
    a retry loses evidence it already had — the failure G10 exists to catch.
    Final-k selection can still drop a prior chunk on score, but that is a
    ranking decision made over the full union rather than an accident of
    discarding the earlier pass.
    """
    seen = {c["chunk_id"] for c in prior}
    return prior + [c for c in fresh if c["chunk_id"] not in seen]
