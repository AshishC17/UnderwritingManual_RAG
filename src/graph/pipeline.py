"""LangGraph StateGraph for the scoped, decomposition-aware RAG pipeline.

    contextualize -> catalog_lenders / conversation_history (metadata only), or
    contextualize -> resolve_scope -> one of:
        clarify        (uncertain scope, or nothing to transform)
        unresolved     (clarification budget spent)
        decline        (not about these manuals)
        reuse_evidence (transform/inspect, stored evidence still in scope)
        decompose -> retrieve -> rerank -> generate

`reuse_evidence` exists because retrieval was never the risk for a pure
reformat request — evidence, once retrieved, does not change. The risk was
`contextualize` composing a fresh question sentence on every turn: two
"identical" rephrase requests, against byte-identical stored evidence,
produced two different completions, because they were two different exact
prompts. `reuse_evidence` never constructs a new question at all — it passes
the user's own words through as `instruction`, so there is nothing left in the
chain that a model rewrote this turn and could have rewritten differently.

Metadata filtering is never delegated to a model. In post_generation mode,
Qwen also reviews the draft against the original request and actual evidence.
Python maps its validated diagnosis to accept, revise, retrieve or clarify;
one recovery attempt is allowed. This review is not an evaluation answer key.

The state is typed rather than a bare dict so the intermediate artifacts are
inspectable: `sub_queries`, `candidates`, and `reranked` all survive into the
final state, which is what makes a failed answer diagnosable without re-running
the pipeline to print things.
"""

from __future__ import annotations

from typing import TypedDict
from copy import deepcopy
import hashlib
import os
import time
from src.guardrails.core import GuardrailError
from src.guardrails.checks import evidence_check, guard_node

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.decompose.decomposer import decompose_query
from src.embed.semantic_cache import embed_query_scoped
from src.embed.sparse import embed_query as embed_query_sparse
from src.generate.generator import (
    generate,
    generate_comparison,
    regenerate_from_evidence,
    revise_answer,
    revise_comparison_answer,
)
from src.rerank.reranker import rerank
from src.decompose.decomposer import reformulate_query
from src.graph import retry as rt
from src.graph.answer_review import (
    review_draft,
    review_comparison_draft,
    request_input,
    JUDGE_MODEL,
)
from src.util.telemetry import observed_node
from src.resolve.followup import classify_followup
from src.resolve.meta import detect_meta_intent, lender_inventory, latest_lender_answer
from src.resolve.scope import (
    ComparisonDecision,
    ScopeDecision,
    resolve_comparison_scopes,
    resolve_query_scope,
    scope_key,
)
from src.store import qdrant_store as qs
from src.store.state_store import evidence_hash

# Behaviors that can be answered from stored evidence alone, if that evidence
# is actually available and still in scope. Everything else — a genuinely new
# question, or a reuse attempt the routing check below rejects — takes the
# full decompose/retrieve/rerank/generate path.
REUSABLE = {"transform_previous_answer", "inspect_previous_answer"}


def _kill_test_pause(node: str) -> None:
    """No-op unless RAG_KILL_TEST_PAUSE names this node.

    Exists only for scripts/verify_kill_points.py, which needs to `kill -9` a
    real server process while it is genuinely inside `retrieve_node` or
    `generate_node` -- not before, not after. Never set in normal operation.
    """
    if os.environ.get("RAG_KILL_TEST_PAUSE") == node:
        time.sleep(30)

# A clarification that has not resolved scope twice will not resolve on a third
# attempt either — the resolver is deterministic, so the same missing signal
# produces the same question. Bounding it turns an infinite loop into an
# honest "I could not determine the scope".
MAX_CLARIFICATION_ROUNDS = 2

CANDIDATES_PER_SUB_QUERY = 15
MAX_RERANK_CANDIDATES = 24
MAX_RERANK_PAIRS = 24
GLOBAL_RERANK_TOKEN_BUDGET = 7500
FINAL_K = 10
COMPARISON_RERANK_CANDIDATES_PER_SCOPE = MAX_RERANK_CANDIDATES // 2
COMPARISON_RERANK_PAIRS_PER_SCOPE = MAX_RERANK_PAIRS // 2
COMPARISON_RERANK_TOKENS_PER_SCOPE = GLOBAL_RERANK_TOKEN_BUDGET // 2
COMPARISON_FINAL_K_PER_SCOPE = FINAL_K // 2

# Pinned once per turn in begin_turn_node, carried unchanged through any
# interrupt/resume (that node does not re-run on resume). Bump this string
# when a change to this file should make an in-flight checkpoint refuse to
# silently resume against code it was not built for.
GRAPH_VERSION = "block-c-2026-09-23"


class CandidateMatch(TypedDict):
    """One sub-query's retrieval relationship to a chunk."""

    sub_query: str
    sub_query_index: int
    rrf_rank: int
    rrf_score: float


class CandidateRecord(TypedDict):
    """A unique chunk plus every sub-query that retrieved it."""

    chunk: dict
    matches: list[CandidateMatch]
    owner_sub_query: str
    owner_sub_query_index: int
    owner_rrf_rank: int


class RerankAssignment(TypedDict):
    """One bounded query-chunk pair sent to the cross-encoder."""

    chunk: dict
    sub_query: str
    sub_query_index: int
    rrf_rank: int
    is_primary: bool


class RAGState(TypedDict, total=False):
    """State threaded through the graph.

    Every intermediate is kept deliberately: `retrieved_candidates` contains
    the complete unique retrieval pool and every sub-query association;
    `candidates` is the globally bounded rerank shortlist; and `reranked` is the
    final scored order. Together they show where a chunk was lost without
    re-running the pipeline.
    """

    turn_id: str
    graph_version: str
    # Concrete collection + immutable manifest hash, not a moving alias.
    index_binding: dict
    question: str
    resolved_question: str
    clarification_rounds: int
    clarifications: list[dict]
    history: list[dict]
    prior_scope: ScopeDecision | None
    scope_inherited: bool
    behavior: str
    instruction: str
    answer_basis: dict
    antecedent_clarification: str | None
    evidence_reused: bool
    scope: ScopeDecision
    metadata_filter: dict | None
    sub_queries: list[str]
    retrieved_candidates: dict[str, CandidateRecord]
    candidates: dict[str, CandidateRecord]
    rerank_assignments: list[RerankAssignment]
    reranked: list[tuple[dict, float]]
    rerank_details: list[dict]
    context_chunks: list[dict]
    answer: str
    # --- corrective retrieval (only populated when max_retries > 0) ---
    assessment: dict
    assessed_action: str
    retry_attempt: int
    max_retries: int
    tried_actions: list[str]
    last_action: str
    last_action_description: str
    recovery_seed: str
    recovery_exhausted: bool
    merge_lost_chunk_ids: list[str]
    reselect_shift: int
    attempt_log: list[dict]
    correction_mode: str
    review_model: str
    review_result: dict
    review_log: list[dict]
    review_clarification: str | None
    correction_error: str | None
    final_status: str
    node_events: list[dict]
    usage_summary: dict
    # --- two-source comparison path ---
    comparison_resolution: ComparisonDecision
    comparison_scopes: list[ScopeDecision]
    comparison_filters: dict[str, dict]
    comparison_sub_queries: dict[str, list[str]]
    comparison_retrieved_candidates: dict[str, dict[str, CandidateRecord]]
    comparison_candidates: dict[str, dict[str, CandidateRecord]]
    comparison_rerank_assignments: dict[str, list[RerankAssignment]]
    comparison_reranked: dict[str, list[tuple[dict, float]]]
    comparison_rerank_details: dict[str, list[dict]]
    comparison_context: dict[str, list[dict]]
    comparison_assessment: dict
    comparison_retry_targets: list[str]


def begin_turn_node(state: RAGState, mode: str, budget: int, review_model: str) -> RAGState:
    """Keep conversational memory; reset all per-turn evidence and diagnostics.

    `turn_id` is deliberately not touched here -- it is set once, by the
    caller, before this node ever runs, and must survive unchanged through
    this reset. `graph_version` is pinned defensively (`state.get(...) or
    GRAPH_VERSION`) rather than always overwritten: if this node somehow
    re-runs on a resumed turn, a version pinned at the turn's real start
    must not get silently replaced by whatever code happens to be running
    now.
    """
    return {"graph_version": state.get("graph_version") or GRAPH_VERSION,
            "resolved_question": state["question"], "scope": {}, "metadata_filter": None,
            "answer": "", "sub_queries": [], "retrieved_candidates": {}, "candidates": {},
            "rerank_assignments": [], "reranked": [], "rerank_details": [], "context_chunks": [],
            "clarifications": [], "clarification_rounds": 0, "scope_inherited": False,
            "behavior": "", "instruction": "", "answer_basis": {},
            "antecedent_clarification": None, "evidence_reused": False,
            "assessment": {}, "assessed_action": "", "retry_attempt": 0, "max_retries": budget,
            "tried_actions": [], "last_action": "", "last_action_description": "", "recovery_seed": "",
            "recovery_exhausted": False, "merge_lost_chunk_ids": [], "reselect_shift": 0, "attempt_log": [],
            "correction_mode": mode, "review_model": review_model, "review_result": {}, "review_log": [],
            "review_clarification": None, "correction_error": None, "final_status": "not_reviewed",
            "node_events": [], "usage_summary": {},
            "comparison_resolution": {}, "comparison_scopes": [], "comparison_filters": {},
            "comparison_sub_queries": {}, "comparison_retrieved_candidates": {},
            "comparison_candidates": {}, "comparison_rerank_assignments": {},
            "comparison_reranked": {}, "comparison_rerank_details": {},
            "comparison_context": {}, "comparison_assessment": {},
            "comparison_retry_targets": []}


def contextualize_node(state: RAGState) -> RAGState:
    """Classify the follow-up, and only compose new question text when a new
    search actually needs one.

    No-ops on the first turn of a conversation, so the single-turn path — every
    eval case — reaches the resolver exactly as it did before this node existed.

    `evidence_reused` is reset to False here unconditionally, since this node
    runs first on every turn: only `reuse_evidence_node` sets it True, and
    without a reset here a prior turn's True would leak into a turn that took
    the full-pipeline path.

    `antecedent_clarification` is a deterministic backstop, not trust in the
    classifier: if it claims "transform" or "inspect" with no history to
    transform, route to clarify regardless of what it said. A router cannot
    set state — see `_route_after_scope` — so this must happen in a node.
    """
    question = state["question"]
    history = state.get("history") or []
    meta_intent = detect_meta_intent(question)
    if meta_intent:
        return {"resolved_question": question, "scope_inherited": False,
                "behavior": meta_intent, "instruction": "",
                "antecedent_clarification": None, "evidence_reused": False}
    result = classify_followup(question, history, prior_scope=state.get("prior_scope"))
    behavior = result["behavior"]

    # The backstop: no history means there is nothing to transform or inspect,
    # regardless of what the classifier predicted. Overriding here, rather
    # than trusting the label, is the same "model proposes, code validates"
    # pattern the scope resolver uses against the manifest.
    if behavior in REUSABLE and not history:
        behavior = "clarify"

    antecedent_clarification = (
        "There's no earlier answer in this conversation for me to work with "
        f"— what would you like me to {question.strip() or 'help with'}?"
        if behavior == "clarify"
        else None
    )

    # `standalone` always becomes `resolved_question` — scope resolution
    # needs it regardless of behavior, and if a reuse attempt turns out not
    # to apply (evidence has gone out of scope), it is what `decompose` falls
    # back to searching with. `instruction`, not this, is what the reuse path
    # actually sends to the generator — that separation is the whole point.
    resolved_question = result["standalone"]
    inherited_index = {}
    if behavior in REUSABLE and history and state.get("index_binding"):
        previous_binding = history[-1].get("index_binding")
        if previous_binding:
            from src.store.index_builds import validate_binding
            validate_binding(previous_binding)
            inherited_index = {"index_binding": previous_binding}
        else:
            # A legacy turn has IDs but no collection identity. Do not hydrate
            # them from an unrelated active build and call it the same evidence.
            behavior = "clarify"
            antecedent_clarification = "That older answer has no saved index-build identity. Please restate the policy question so I can check the current manual."
    return {
        **inherited_index,
        "resolved_question": resolved_question,
        "scope_inherited": resolved_question != question,
        "behavior": behavior,
        "instruction": result["instruction"],
        "antecedent_clarification": antecedent_clarification,
        "evidence_reused": False,
    }


def _route_after_contextualize(state: RAGState) -> str:
    if state.get("behavior") == "list_corpus_lenders":
        return "catalog_lenders"
    if state.get("behavior") == "last_lender_answer":
        return "conversation_history"
    if state.get("behavior") == "compare_policy":
        return "resolve_comparison_scopes"
    return "resolve_scope"


def catalog_lenders_node(state: RAGState) -> RAGState:
    """Read corpus identity from the manifest, not from policy chunks."""
    return {"scope": {"status": "not_applicable", "reason": "corpus_inventory"},
            "answer": lender_inventory(),
            "answer_basis": {"kind": "corpus_manifest", "source": "config/corpus_manifest.json"},
            "final_status": "deterministic"}


def conversation_history_node(state: RAGState) -> RAGState:
    """Answer a question about this thread from turns saved in its checkpoint."""
    answer, basis = latest_lender_answer(state.get("history") or [])
    return {"scope": {"status": "not_applicable", "reason": "conversation_history"},
            "answer": answer, "answer_basis": basis,
            "final_status": "deterministic"}


def resolve_scope_node(state: RAGState) -> RAGState:
    """Resolve lender/version/date, then materialize the exact Qdrant filter.

    Resolution runs against `resolved_question` — the original question plus
    whatever clarifications have been folded into it — so a second pass after
    a clarification sees the added lender or version. On the first pass the
    two are identical.
    """
    # Scope comes only from the question text. Carrying the previous lender
    # forward is the rewriter's job, because only it can tell an omission
    # ("what about the ceiling?") from a switch ("what about the other lender?").
    # A deterministic fallback here could not: it saw `missing_lender` in both
    # cases and pinned the previous lender onto a question explicitly asking to
    # leave it. Two mechanisms that can disagree about scope is worse than one.
    question = state.get("resolved_question") or state["question"]
    decision = resolve_query_scope(question)

    if decision["status"] != "resolved":
        return {
            "scope": decision,
            "metadata_filter": None,
            "resolved_question": question,
        }

    query_filter = qs.scope_filter(**decision["filter_args"])
    return {
        "scope": decision,
        "resolved_question": question,
        "metadata_filter": (
            query_filter.model_dump(mode="json", exclude_none=True)
            if query_filter is not None
            else None
        ),
    }


def resolve_comparison_scopes_node(state: RAGState) -> RAGState:
    """Resolve and materialize exactly two independent Qdrant scope filters."""
    question = state.get("resolved_question") or state["question"]
    decision = resolve_comparison_scopes(
        question,
        prior_scope=state.get("prior_scope"),
    )
    if decision.get("status") != "resolved":
        return {
            "comparison_resolution": decision,
            "comparison_scopes": [],
            "comparison_filters": {},
            "resolved_question": question,
        }

    scopes = decision["scopes"]
    filters = {}
    for scope in scopes:
        query_filter = qs.scope_filter(**scope["filter_args"])
        filters[scope_key(scope)] = (
            query_filter.model_dump(mode="json", exclude_none=True)
            if query_filter is not None else {}
        )
    return {
        "comparison_resolution": decision,
        "comparison_scopes": scopes,
        "comparison_filters": filters,
        "resolved_question": question,
        # Ordinary scope is deliberately not overloaded with two authorities.
        "scope": {},
        "metadata_filter": None,
    }


def _route_after_comparison_scope(state: RAGState) -> str:
    if (state.get("comparison_resolution") or {}).get("status") == "resolved":
        return "plan_comparison"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        return "unresolved"
    return "clarify"


def _route_after_clarify(state: RAGState) -> str:
    return (
        "resolve_comparison_scopes"
        if state.get("behavior") == "compare_policy"
        else "resolve_scope"
    )


def _route_after_scope(state: RAGState) -> str:
    # Checked first, ahead of scope status: an antecedent-missing or
    # out-of-scope request can arrive alongside a scope that resolves fine
    # (or fails for an unrelated reason) — neither should let it fall through
    # into a search it was never asking for.
    if state.get("antecedent_clarification"):
        return "clarify"
    if state.get("behavior") == "decline_out_of_scope":
        return "decline"

    if state["scope"]["status"] == "resolved":
        if state.get("behavior") in REUSABLE and _reuse_is_valid(state):
            return "reuse_evidence"
        return "decompose"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        return "unresolved"
    return "clarify"


def _reuse_is_valid(state: RAGState) -> bool:
    """Is there stored evidence, in the scope currently in force, to reuse?

    Both checks matter. No history means nothing was ever retrieved on this
    thread. A scope mismatch means the stored evidence belongs to a manual
    the user has since left — exactly the case that started this design: a
    lender switch must not silently reuse the old lender's evidence.
    """
    history = state.get("history")
    if not history:
        return False
    prior = state.get("prior_scope")
    scope = state["scope"]
    return bool(
        prior
        and prior.get("lender_id") == scope.get("lender_id")
        and prior.get("document_version") == scope.get("document_version")
    )


def clarify_node(state: RAGState) -> RAGState:
    """Pause the graph and ask the user, rather than guessing.

    Two distinct things route here: uncertain scope (the resolver's own
    question) and a missing antecedent (nothing in this conversation for a
    "transform" or "inspect" request to act on). `antecedent_clarification`
    is checked first since it is the more specific case when both happen to
    be present.

    `interrupt` raises on the first pass, handing its payload to the caller.
    Resuming with `Command(resume=<reply>)` re-enters this node from the top
    and returns the reply here instead of raising.

    That re-execution is why nothing may precede the `interrupt` call: the
    round counter is incremented below it, so one clarification counts once
    rather than twice.
    """
    antecedent = state.get("antecedent_clarification")
    comparison_clarification = (state.get("comparison_resolution") or {}).get("clarification_question")
    question = (
        state.get("review_clarification")
        or antecedent
        or comparison_clarification
        or state["scope"]["clarification_question"]
    )
    reason = (state.get("review_result", {}).get("reason") if state.get("review_clarification")
              else "missing_antecedent" if antecedent
              else (state.get("comparison_resolution") or {}).get("reason")
              or state["scope"].get("reason"))

    reply = interrupt({"kind": "clarification", "question": question, "reason": reason})

    base = state.get("resolved_question") or state["question"]
    return {
        # Folded into one complete question rather than kept as a separate
        # turn: the clarification carries no information need of its own.
        "resolved_question": f"{base} {reply}".strip(),
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        "clarifications": [
            *state.get("clarifications", []),
            {"ask": question, "reply": str(reply)},
        ],
        "review_clarification": None,
        "antecedent_clarification": None,
    }


def unresolved_node(state: RAGState) -> RAGState:
    """Give up honestly after the clarification budget is spent."""
    return {
        "sub_queries": [],
        "retrieved_candidates": {},
        "candidates": {},
        "rerank_assignments": [],
        "reranked": [],
        "rerank_details": [],
        "context_chunks": [],
        "answer": (
            "I could not determine the two manuals to compare after "
            f"{state.get('clarification_rounds', 0)} attempts. Name both lenders "
            "and any required versions explicitly."
            if state.get("behavior") == "compare_policy"
            else "I could not determine which manual to use after "
            f"{state.get('clarification_rounds', 0)} attempts. Name the lender "
            "and version explicitly — for example 'NovaCred V0' or 'LumenTrail V1'."
        ),
    }


def decline_node(state: RAGState) -> RAGState:
    """A fixed, non-generated decline for requests outside the manuals.

    No model call: what to say does not depend on the specifics of an
    out-of-scope request, and generating a bespoke decline would be spending
    a call to produce something a template says just as well.
    """
    return {
        "sub_queries": [],
        "retrieved_candidates": {},
        "candidates": {},
        "rerank_assignments": [],
        "reranked": [],
        "rerank_details": [],
        "context_chunks": [],
        "answer": (
            "This assistant answers questions about the NovaCred and "
            "LumenTrail underwriting manuals only."
        ),
    }


def reuse_evidence_node(state: RAGState) -> RAGState:
    """Answer a transform/inspect follow-up from the previous turn's stored
    evidence — no decompose, no retrieve, no rerank.

    The chunks are re-hydrated by id from Qdrant, not re-searched: a point
    lookup, no Voyage call, no risk of a fresh search returning a different
    candidate set for what is supposed to be the same evidence.

    The stored turn's own `question` is passed through too, not just its
    answer and chunks: a policy manual states general rules, not
    scenario-specific facts like "these two fired at the same time," and that
    fact lives only in the original question — dropping it is how an
    already-fired result gets misread as later, skippable work.
    """
    client = qs.connect()
    last_turn = state["history"][-1]
    chunks = qs.get_by_chunk_ids(client, last_turn.get("chunk_ids") or [])
    evidence_check(chunks, state.get("scope"))

    if state.get("correction_mode") == "post_generation" and _evidence_fault(
        state, chunks, expected_evidence_hash=last_turn.get("evidence_hash")
    ):
        # Do not let a stale/corrupt prior artifact reach even the draft model.
        return {"answer": "", "context_chunks": chunks, "evidence_reused": True}

    try:
        result = regenerate_from_evidence(
            instruction=state.get("instruction") or state["question"],
            prior_answer=last_turn.get("answer", ""),
            chunks=chunks,
            original_question=last_turn.get("question", ""),
        )
    except Exception as exc:
        if state.get("correction_mode") != "post_generation":
            raise
        return {"answer": "", "context_chunks": chunks, "evidence_reused": True,
                "correction_error": f"reuse_generation:{type(exc).__name__}"}

    return {
        "answer": result["answer"],
        "context_chunks": chunks,
        # Discovered here, not predicted upstream: whether the prior answer
        # needed correcting is only knowable once it has been checked.
        "behavior": (
            "correct_previous_answer" if result["corrected_prior_claim"] else state.get("behavior")
        ),
        "evidence_reused": True,
        "sub_queries": [],
        "retrieved_candidates": {},
        "candidates": {},
        "rerank_assignments": [],
        "reranked": [],
        "rerank_details": [],
    }


def decompose_node(state: RAGState) -> RAGState:
    """Split the question if it asks 3+ distinct things, else pass it through."""
    return {"sub_queries": decompose_query(state["resolved_question"])}


def _chunk_token_cost(chunk: dict) -> int:
    """Use ingestion's token count, with the reranker's fallback estimate."""
    return chunk.get("token_count") or max(1, len(chunk["text"]) // 4)


def _pool_candidates(
    sub_queries: list[str],
    hit_lists: list[list],
) -> dict[str, CandidateRecord]:
    """Deduplicate chunk payloads while preserving all query relationships.

    Ownership is assigned only after all searches finish. A chunk is owned by
    the sub-query where it had its best RRF rank, rather than whichever query
    happened to run first. Ties use sub-query order solely for determinism.
    """
    pool: dict[str, CandidateRecord] = {}
    for sub_query_index, (sub_query, hits) in enumerate(zip(sub_queries, hit_lists)):
        for rrf_rank, hit in enumerate(hits, start=1):
            chunk = dict(hit.payload)
            chunk_id = chunk["chunk_id"]
            match: CandidateMatch = {
                "sub_query": sub_query,
                "sub_query_index": sub_query_index,
                "rrf_rank": rrf_rank,
                "rrf_score": float(hit.score or 0.0),
            }
            if chunk_id not in pool:
                pool[chunk_id] = {
                    "chunk": chunk,
                    "matches": [],
                    # Filled after all sub-query results have been collected.
                    "owner_sub_query": "",
                    "owner_sub_query_index": 0,
                    "owner_rrf_rank": 0,
                }
            pool[chunk_id]["matches"].append(match)

    for record in pool.values():
        owner = min(
            record["matches"],
            key=lambda match: (match["rrf_rank"], match["sub_query_index"]),
        )
        record["owner_sub_query"] = owner["sub_query"]
        record["owner_sub_query_index"] = owner["sub_query_index"]
        record["owner_rrf_rank"] = owner["rrf_rank"]

    return pool


def _select_rerank_candidates(
    pool: dict[str, CandidateRecord],
    max_candidates: int = MAX_RERANK_CANDIDATES,
    max_tokens: int = GLOBAL_RERANK_TOKEN_BUDGET,
) -> dict[str, CandidateRecord]:
    """Build one fair, globally bounded shortlist of unique chunks.

    Ordering by each chunk's best RRF rank interleaves the sub-query result
    lists: all rank-1 owners precede rank-2 owners, and so on. This avoids the
    first sub-query consuming the budget. Each selected chunk is later reranked
    exactly once against its best-ranked owner.
    """
    ordered = sorted(
        pool.items(),
        key=lambda item: (
            item[1]["owner_rrf_rank"],
            item[1]["owner_sub_query_index"],
            item[0],
        ),
    )

    selected: dict[str, CandidateRecord] = {}
    used_tokens = 0
    for chunk_id, record in ordered:
        if len(selected) >= max_candidates:
            break
        cost = _chunk_token_cost(record["chunk"])
        if used_tokens + cost > max_tokens:
            continue
        selected[chunk_id] = record
        used_tokens += cost

    return selected


def _build_rerank_assignments(
    candidates: dict[str, CandidateRecord],
    max_pairs: int = MAX_RERANK_PAIRS,
    max_tokens: int = GLOBAL_RERANK_TOKEN_BUDGET,
) -> list[RerankAssignment]:
    """Assign every unique chunk once, then use spare budget to resolve ties.

    The primary assignment is the chunk's best-RRF owner. If multiple
    sub-queries found the same chunk at that exact best rank, first-owner
    tie-breaking would again discard potentially useful intent information.
    Such tied associations are scored only when capacity remains inside the
    same global pair and token budgets; this never expands into all N×K pairs.
    """
    assignments: list[RerankAssignment] = []
    used_tokens = 0

    for record in candidates.values():
        chunk = record["chunk"]
        assignments.append({
            "chunk": chunk,
            "sub_query": record["owner_sub_query"],
            "sub_query_index": record["owner_sub_query_index"],
            "rrf_rank": record["owner_rrf_rank"],
            "is_primary": True,
        })
        used_tokens += _chunk_token_cost(chunk)

    tied_alternates = []
    for chunk_id, record in candidates.items():
        for match in record["matches"]:
            if (
                match["rrf_rank"] == record["owner_rrf_rank"]
                and match["sub_query_index"] != record["owner_sub_query_index"]
            ):
                tied_alternates.append((
                    record["owner_rrf_rank"],
                    match["sub_query_index"],
                    chunk_id,
                    record,
                    match,
                ))

    for _, _, _, record, match in sorted(tied_alternates):
        if len(assignments) >= max_pairs:
            break
        cost = _chunk_token_cost(record["chunk"])
        if used_tokens + cost > max_tokens:
            continue
        assignments.append({
            "chunk": record["chunk"],
            "sub_query": match["sub_query"],
            "sub_query_index": match["sub_query_index"],
            "rrf_rank": match["rrf_rank"],
            "is_primary": False,
        })
        used_tokens += cost

    return assignments


def _score_rerank_assignments(
    assignments: list[RerankAssignment],
    candidates: dict[str, CandidateRecord],
    *,
    max_tokens: int = GLOBAL_RERANK_TOKEN_BUDGET,
) -> tuple[list[tuple[dict, float]], list[dict]]:
    """Shared bounded cross-encoder scoring for ordinary and comparison paths."""
    by_sub_query: dict[str, list[RerankAssignment]] = {}
    for assignment in assignments:
        by_sub_query.setdefault(assignment["sub_query"], []).append(assignment)

    scored: dict[str, tuple[dict, float]] = {}
    score_details: dict[str, list[dict]] = {}
    for sub_query, grouped in by_sub_query.items():
        chunks = [assignment["chunk"] for assignment in grouped]
        for chunk, score in rerank(sub_query, chunks, max_tokens=max_tokens):
            chunk_id = chunk["chunk_id"]
            assignment = next(
                item for item in grouped if item["chunk"]["chunk_id"] == chunk_id
            )
            score_details.setdefault(chunk_id, []).append({
                "sub_query": sub_query,
                "sub_query_index": assignment["sub_query_index"],
                "rrf_rank": assignment["rrf_rank"],
                "rerank_score": score,
                "is_primary": assignment["is_primary"],
            })
            if chunk_id not in scored or score > scored[chunk_id][1]:
                scored[chunk_id] = (chunk, score)

    ranked = sorted(scored.values(), key=lambda pair: pair[1], reverse=True)
    details = []
    for final_rank, (chunk, score) in enumerate(ranked, start=1):
        record = candidates[chunk["chunk_id"]]
        winning_score = max(
            score_details[chunk["chunk_id"]],
            key=lambda item: item["rerank_score"],
        )
        details.append({
            "chunk_id": chunk["chunk_id"],
            "final_rank": final_rank,
            "rerank_score": score,
            "owner_sub_query": record["owner_sub_query"],
            "owner_sub_query_index": record["owner_sub_query_index"],
            "owner_rrf_rank": record["owner_rrf_rank"],
            "winning_sub_query": winning_score["sub_query"],
            "winning_sub_query_index": winning_score["sub_query_index"],
            "matches": record["matches"],
            "rerank_scores": score_details[chunk["chunk_id"]],
        })
    return ranked, details


def retrieve_node(state: RAGState) -> RAGState:
    """Hybrid-search each sub-query, retain associations, then bound globally.

    Each sub-query searches at full depth rather than a reduced one: a
    decomposed sub-query is just a normal focused query, so it earns the same
    treatment any single query gets, and shrinking depth per sub-query would
    introduce a second parameter to tune with no evidence it is needed.

    Dedup is on exact `chunk_id`, not embedding similarity. Unlike first-wins
    dedup, the pool retains every sub-query and RRF rank that found the chunk.
    The rerank shortlist is capped across the whole decomposed question, so
    three sub-queries cannot silently triple the expensive reranker budget.
    """
    _kill_test_pause("retrieve")
    client = qs.connect()
    query_filter = qs.scope_filter(**state["scope"]["filter_args"])
    hit_lists = []

    for sub_query in state["sub_queries"]:
        hit_lists.append(
            qs.search_hybrid(
                client,
                embed_query_scoped(
                    sub_query,
                    state["scope"]["filter_args"]["lender_id"],
                    state["scope"]["filter_args"]["document_version"],
                ),
                embed_query_sparse(sub_query),
                limit=CANDIDATES_PER_SUB_QUERY,
                prefetch_limit=CANDIDATES_PER_SUB_QUERY * 2,
                query_filter=query_filter,
            )
        )

    pool = _pool_candidates(state["sub_queries"], hit_lists)
    candidates = _select_rerank_candidates(pool)
    assignments = _build_rerank_assignments(candidates)
    return {
        "retrieved_candidates": pool,
        "candidates": candidates,
        "rerank_assignments": assignments,
    }


def rerank_node(state: RAGState) -> RAGState:
    """Score each unique chunk once against its best-RRF sub-query.

    Reranking the pooled set against the *original* compound question would
    reintroduce exactly the dilution decomposition exists to remove — a
    cross-encoder still has to split attention across clauses. Scoring against
    the originating sub-query keeps each judgement focused.

    One rerank call per represented sub-query, not per chunk: the cross-encoder
    takes a batch. The shortlist was bounded globally before grouping, so the
    sum of all batches is at most MAX_RERANK_CANDIDATES and
    GLOBAL_RERANK_TOKEN_BUDGET rather than that budget per sub-query.
    """
    ranked, details = _score_rerank_assignments(
        state["rerank_assignments"],
        state["candidates"],
    )

    return {
        "reranked": ranked,
        "rerank_details": details,
        "context_chunks": _select_final_context(ranked, state.get("reselect_shift", 0)),
    }


def plan_comparison_node(state: RAGState) -> RAGState:
    """Decompose once, then use the same information needs under both scopes."""
    queries = decompose_query(state["resolved_question"])
    by_scope = {scope_key(scope): list(queries) for scope in state["comparison_scopes"]}
    return {
        "comparison_sub_queries": by_scope,
        # Retain the ordinary diagnostic field without duplicating query text.
        "sub_queries": list(queries),
    }


def retrieve_comparison_node(state: RAGState) -> RAGState:
    """Hybrid-search both scopes independently under a shared expensive budget."""
    client = qs.connect()
    pools: dict[str, dict[str, CandidateRecord]] = {}
    candidates_by_scope: dict[str, dict[str, CandidateRecord]] = {}
    assignments_by_scope: dict[str, list[RerankAssignment]] = {}

    for scope in state["comparison_scopes"]:
        key = scope_key(scope)
        query_filter = qs.scope_filter(**scope["filter_args"])
        queries = state["comparison_sub_queries"][key]
        hit_lists = [
            qs.search_hybrid(
                client,
                embed_query_scoped(q, scope["lender_id"], scope["document_version"]),
                embed_query_sparse(q),
                limit=CANDIDATES_PER_SUB_QUERY,
                prefetch_limit=CANDIDATES_PER_SUB_QUERY * 2,
                query_filter=query_filter,
            )
            for q in queries
        ]
        pool = _pool_candidates(queries, hit_lists)
        candidates = _select_rerank_candidates(
            pool,
            max_candidates=COMPARISON_RERANK_CANDIDATES_PER_SCOPE,
            max_tokens=COMPARISON_RERANK_TOKENS_PER_SCOPE,
        )
        pools[key] = pool
        candidates_by_scope[key] = candidates
        assignments_by_scope[key] = _build_rerank_assignments(
            candidates,
            max_pairs=COMPARISON_RERANK_PAIRS_PER_SCOPE,
            max_tokens=COMPARISON_RERANK_TOKENS_PER_SCOPE,
        )

    return {
        "comparison_retrieved_candidates": pools,
        "comparison_candidates": candidates_by_scope,
        "comparison_rerank_assignments": assignments_by_scope,
    }


def rerank_comparison_node(state: RAGState) -> RAGState:
    """Rerank within each source boundary and select equal initial context slots."""
    ranked_by_scope: dict[str, list[tuple[dict, float]]] = {}
    details_by_scope: dict[str, list[dict]] = {}
    contexts: dict[str, list[dict]] = {}
    flattened_ranked: list[tuple[dict, float]] = []

    for scope in state["comparison_scopes"]:
        key = scope_key(scope)
        ranked, details = _score_rerank_assignments(
            state["comparison_rerank_assignments"].get(key, []),
            state["comparison_candidates"].get(key, {}),
            max_tokens=COMPARISON_RERANK_TOKENS_PER_SCOPE,
        )
        ranked_by_scope[key] = ranked
        details_by_scope[key] = details
        contexts[key] = [chunk for chunk, _ in ranked[:COMPARISON_FINAL_K_PER_SCOPE]]
        flattened_ranked.extend(ranked)

    flattened_context = [
        chunk
        for scope in state["comparison_scopes"]
        for chunk in contexts[scope_key(scope)]
    ]
    return {
        "comparison_reranked": ranked_by_scope,
        "comparison_rerank_details": details_by_scope,
        "comparison_context": contexts,
        "context_chunks": flattened_context,
        # Compatibility for persistence/debug tooling; never interpreted as one
        # global ranking because scores come from different scope/query batches.
        "reranked": flattened_ranked,
    }


def generate_comparison_node(state: RAGState) -> RAGState:
    try:
        if state.get("retry_attempt", 0):
            answer = revise_comparison_answer(
                request_input(state),
                state.get("answer", ""),
                state["comparison_scopes"],
                state["comparison_context"],
                state.get("review_result", {}),
            )
        else:
            answer = generate_comparison(
                state["resolved_question"],
                state["comparison_scopes"],
                state["comparison_context"],
            )
        return {"answer": answer}
    except Exception as exc:
        return {"answer": "", "correction_error": f"comparison_generation:{type(exc).__name__}"}


def generate_node(state: RAGState) -> RAGState:
    """Answer from the top-k chunks, against the whole question.

    The user asked the whole question; sub-queries were a retrieval device, not
    a rewrite of what they wanted answered. `resolved_question` is used rather
    than `question` because a clarified lender or version is part of what was
    asked, not scaffolding around it.
    """
    _kill_test_pause("generate")
    if state.get("correction_mode") != "post_generation":
        return {"answer": generate(state["resolved_question"], state["context_chunks"])}
    try:
        if state.get("retry_attempt", 0):
            answer = revise_answer(request_input(state), state.get("answer", ""), state["context_chunks"], state.get("review_result", {}))
        else:
            answer = generate(state["resolved_question"], state["context_chunks"])
        return {"answer": answer}
    except Exception as exc:
        return {"answer": "", "correction_error": f"generation:{type(exc).__name__}"}


def _evidence_fault(state: dict, chunks: list[dict], expected_evidence_hash: str | None = None) -> str | None:
    scope = state.get("scope") or {}
    if scope.get("status") != "resolved":
        return "wrong_scope"
    if any(c.get("lender_id") != scope.get("lender_id") or c.get("document_version") != scope.get("document_version")
           or (scope.get("document_id") and c.get("document_id") != scope["document_id"]) for c in chunks):
        return "wrong_scope"
    if not chunks:
        return "empty_context"
    # Only reuse_evidence_node passes expected_evidence_hash. Historical evidence
    # must not silently become today's policy: if the chunks a prior turn cited
    # were re-chunked or re-embedded since, their ids may now point at different
    # text than what the stored answer was actually built from.
    if expected_evidence_hash and evidence_hash(c["chunk_id"] for c in chunks) != expected_evidence_hash:
        return "stale_evidence"
    return None


def _fault_review(fault: str) -> dict:
    return {"action": "retrieve" if fault == "empty_context" else "stop",
            "reason": fault, "requirements": [], "unsupported_claims": [],
            "citation_issues": [],
            "supported_excerpts": [], "needs_clarification": False, "clarification_question": ""}


def check_evidence_node(state: RAGState) -> RAGState:
    fault = _evidence_fault(state, state.get("context_chunks") or [])
    update = {"assessment": {"status": fault or "scope_checked"}}
    if fault:
        review = _fault_review(fault)
        update.update(review_result=review, review_log=[*state.get("review_log", []),
                      {"attempt": state.get("retry_attempt", 0), "stage": "evidence_guard",
                       "draft": "", "context": _context_fingerprints(state), "review": review,
                       "selected_action": _correction_route({**state, "review_result": review})}])
    return update


def _comparison_evidence_fault(state: RAGState) -> tuple[str | None, list[str]]:
    scopes = state.get("comparison_scopes") or []
    contexts = state.get("comparison_context") or {}
    if len(scopes) != 2:
        return "wrong_comparison_scope", []
    expected = {scope_key(scope): scope for scope in scopes}
    if set(contexts) != set(expected):
        return "wrong_comparison_scope", []
    empty = []
    for key, scope in expected.items():
        chunks = contexts[key]
        if not chunks:
            empty.append(key)
            continue
        if any(
            chunk.get("lender_id") != scope.get("lender_id")
            or chunk.get("document_version") != scope.get("document_version")
            or chunk.get("document_id") != scope.get("document_id")
            for chunk in chunks
        ):
            return "wrong_scope", [key]
    return ("empty_comparison_side", empty) if empty else (None, [])


def check_comparison_evidence_node(state: RAGState) -> RAGState:
    fault, targets = _comparison_evidence_fault(state)
    assessment = {
        "status": fault or "comparison_scopes_checked",
        "retry_scope_keys": targets,
    }
    update: RAGState = {"comparison_assessment": assessment, "assessment": assessment}
    if fault:
        review = {**_fault_review(fault), "missing_scope_keys": targets}
        update.update(
            review_result=review,
            review_log=[
                *state.get("review_log", []),
                {
                    "attempt": state.get("retry_attempt", 0),
                    "stage": "comparison_evidence_guard",
                    "draft": "",
                    "context": _context_fingerprints(state),
                    "review": review,
                    "selected_action": _comparison_correction_route(
                        {**state, "review_result": review}
                    ),
                },
            ],
        )
    return update


def _context_fingerprints(state: dict) -> list[dict]:
    return [{"chunk_id": c["chunk_id"], "sha256": hashlib.sha256(c["text"].encode()).hexdigest()}
            for c in state.get("context_chunks", [])]


def _correction_route(state: RAGState) -> str:
    action = (state.get("review_result") or {}).get("action", "stop")
    if action == "clarify" and state.get("clarification_rounds", 0) < MAX_CLARIFICATION_ROUNDS:
        return "clarify"
    if state.get("retry_attempt", 0) >= state.get("max_retries", 0):
        return "finalize"
    return {"revise": "revise_answer", "retrieve": "targeted_retrieve"}.get(action, "finalize")


def _comparison_correction_route(state: RAGState) -> str:
    action = (state.get("review_result") or {}).get("action", "stop")
    if action == "clarify" and state.get("clarification_rounds", 0) < MAX_CLARIFICATION_ROUNDS:
        return "clarify"
    if state.get("retry_attempt", 0) >= state.get("max_retries", 0):
        return "finalize"
    return {
        "revise": "revise_comparison",
        "retrieve": "targeted_comparison_retrieve",
    }.get(action, "finalize")


def _after_evidence(state: RAGState) -> str:
    if state["assessment"]["status"] == "scope_checked":
        return "generate"
    return _correction_route(state)


def _after_comparison_evidence(state: RAGState) -> str:
    if state["comparison_assessment"]["status"] == "comparison_scopes_checked":
        return "generate_comparison"
    return _comparison_correction_route(state)


def review_answer_node(state: RAGState) -> RAGState:
    fault = _evidence_fault(state, state.get("context_chunks") or [])
    if state.get("correction_error"):
        result = _fault_review(state["correction_error"])
    elif fault:
        result = _fault_review(fault)
    elif not state.get("answer", "").strip():
        result = {**_fault_review("empty_draft"), "action": "revise"}
    else:
        try:
            result = review_draft(state, model=state["review_model"])
        except Exception as exc:
            result = _fault_review(f"review_error:{type(exc).__name__}")
    updated = {**state, "review_result": result}
    selected = _correction_route(updated)
    entry = {"attempt": state.get("retry_attempt", 0), "stage": "answer_review", "draft": state.get("answer", ""),
             "context": _context_fingerprints(state),
             "review": result, "selected_action": selected}
    return {"review_result": result, "review_log": [*state.get("review_log", []), entry],
            "review_clarification": result.get("clarification_question") if selected == "clarify" else None}


def revise_answer_node(state: RAGState) -> RAGState:
    update = {"retry_attempt": state.get("retry_attempt", 0) + 1,
              "tried_actions": [*state.get("tried_actions", []), "revise_answer"], "last_action": "revise_answer"}
    try:
        update["answer"] = revise_answer(request_input(state), state["answer"], state["context_chunks"], state["review_result"])
    except Exception as exc:
        update.update(answer="", correction_error=f"revision:{type(exc).__name__}")
    return update


def targeted_retrieve_node(state: RAGState) -> RAGState:
    """One scoped search. Preserve the pool; reranking retains its global caps."""
    update = {"retry_attempt": state.get("retry_attempt", 0) + 1,
              "tried_actions": [*state.get("tried_actions", []), "targeted_retrieve"], "last_action": "targeted_retrieve"}
    needs = [r["need"] for r in state.get("review_result", {}).get("requirements", []) if r["evidence"] == "missing"]
    query = state["resolved_question"] + ("\nFocus on evidence for: " + "; ".join(needs[:3]) if needs else "")
    pool = deepcopy(state.get("retrieved_candidates") or state.get("candidates") or {})
    # Reused evidence may never have had a search-pool record this turn.
    for rank, chunk in enumerate(state.get("context_chunks", []), 1):
        if chunk["chunk_id"] not in pool:
            match = {"sub_query": state["resolved_question"], "sub_query_index": 0, "rrf_rank": rank, "rrf_score": 0.0}
            pool[chunk["chunk_id"]] = {"chunk": chunk, "matches": [match], "owner_sub_query": match["sub_query"],
                                       "owner_sub_query_index": 0, "owner_rrf_rank": rank}
    prior_ids = list(pool)
    queries = list(state.get("sub_queries") or [state["resolved_question"]])
    query_index = len(queries)
    update["recovery_seed"] = query
    try:
        client = qs.connect()
        query_filter = qs.scope_filter(**state["scope"]["filter_args"])
        hits = qs.search_hybrid(
            client,
            embed_query_scoped(
                query,
                state["scope"]["filter_args"]["lender_id"],
                state["scope"]["filter_args"]["document_version"],
            ),
            embed_query_sparse(query),
            limit=CANDIDATES_PER_SUB_QUERY * 2, prefetch_limit=CANDIDATES_PER_SUB_QUERY * 4,
            query_filter=query_filter)
        fresh = _pool_candidates([query], [hits])
        if _evidence_fault(state, [r["chunk"] for r in fresh.values()]) == "wrong_scope":
            raise ValueError("retrieval returned out-of-scope evidence")
        for cid, record in fresh.items():
            for match in record["matches"]:
                match["sub_query_index"] = query_index
            if cid in pool:
                record["matches"] = pool[cid]["matches"] + record["matches"]
            owner = min(record["matches"], key=lambda m: (m["rrf_rank"], m["sub_query_index"]))
            record.update(owner_sub_query=owner["sub_query"], owner_sub_query_index=owner["sub_query_index"], owner_rrf_rank=owner["rrf_rank"])
            pool[cid] = record
        candidates = _select_rerank_candidates(pool)
        update.update(retrieved_candidates=pool, candidates=candidates, sub_queries=queries + [query],
                      rerank_assignments=_build_rerank_assignments(candidates),
                      merge_lost_chunk_ids=rt.check_merge_invariant(prior_ids, list(pool)))
    except Exception as exc:
        update.update(correction_error=f"retrieval:{type(exc).__name__}", review_result=_fault_review("recovery_failed"))
    return update


def review_comparison_node(state: RAGState) -> RAGState:
    fault, targets = _comparison_evidence_fault(state)
    if state.get("correction_error"):
        result = {**_fault_review(state["correction_error"]), "missing_scope_keys": targets}
    elif fault:
        result = {**_fault_review(fault), "missing_scope_keys": targets}
    elif not state.get("answer", "").strip():
        result = {**_fault_review("empty_comparison_draft"), "action": "revise",
                  "missing_scope_keys": []}
    else:
        try:
            result = review_comparison_draft(state, model=state["review_model"])
        except Exception as exc:
            result = {**_fault_review(f"review_error:{type(exc).__name__}"),
                      "missing_scope_keys": []}
    updated = {**state, "review_result": result}
    selected = _comparison_correction_route(updated)
    entry = {
        "attempt": state.get("retry_attempt", 0),
        "stage": "comparison_answer_review",
        "draft": state.get("answer", ""),
        "context": _context_fingerprints(state),
        "review": result,
        "selected_action": selected,
    }
    return {
        "review_result": result,
        "review_log": [*state.get("review_log", []), entry],
        "review_clarification": (
            result.get("clarification_question") if selected == "clarify" else None
        ),
    }


def revise_comparison_node(state: RAGState) -> RAGState:
    update: RAGState = {
        "retry_attempt": state.get("retry_attempt", 0) + 1,
        "tried_actions": [*state.get("tried_actions", []), "revise_comparison"],
        "last_action": "revise_comparison",
    }
    try:
        update["answer"] = revise_comparison_answer(
            request_input(state),
            state["answer"],
            state["comparison_scopes"],
            state["comparison_context"],
            state["review_result"],
        )
    except Exception as exc:
        update.update(answer="", correction_error=f"comparison_revision:{type(exc).__name__}")
    return update


def targeted_comparison_retrieve_node(state: RAGState) -> RAGState:
    """Deepen only reviewer-identified incomplete scopes; preserve both pools."""
    update: RAGState = {
        "retry_attempt": state.get("retry_attempt", 0) + 1,
        "tried_actions": [*state.get("tried_actions", []), "targeted_comparison_retrieve"],
        "last_action": "targeted_comparison_retrieve",
    }
    allowed = {scope_key(scope): scope for scope in state.get("comparison_scopes", [])}
    targets = list(dict.fromkeys(
        (state.get("review_result") or {}).get("missing_scope_keys")
        or (state.get("comparison_assessment") or {}).get("retry_scope_keys")
        or []
    ))
    if not targets or any(key not in allowed for key in targets):
        return {
            **update,
            "correction_error": "comparison_retrieval:no_valid_retry_scope",
            "review_result": _fault_review("recovery_failed"),
        }

    pools = deepcopy(state.get("comparison_retrieved_candidates") or {})
    candidates_by_scope = deepcopy(state.get("comparison_candidates") or {})
    assignments_by_scope = deepcopy(state.get("comparison_rerank_assignments") or {})
    queries_by_scope = deepcopy(state.get("comparison_sub_queries") or {})
    lost: list[str] = []
    client = qs.connect()

    try:
        for key in targets:
            scope = allowed[key]
            needs = [
                requirement["need"]
                for requirement in (state.get("review_result") or {}).get("requirements", [])
                if requirement.get("evidence") == "missing"
                and requirement.get("scope_key") == key
            ]
            query = state["resolved_question"] + (
                "\nFocus only on missing evidence for this source: " + "; ".join(needs[:3])
                if needs else "\nRetrieve additional evidence for this source."
            )
            query_index = len(queries_by_scope.get(key, []))
            queries_by_scope.setdefault(key, []).append(query)
            query_filter = qs.scope_filter(**scope["filter_args"])
            hits = qs.search_hybrid(
                client,
                embed_query_scoped(query, scope["lender_id"], scope["document_version"]),
                embed_query_sparse(query),
                limit=CANDIDATES_PER_SUB_QUERY * 2,
                prefetch_limit=CANDIDATES_PER_SUB_QUERY * 4,
                query_filter=query_filter,
            )
            prior_pool = pools.get(key, {})
            prior_ids = list(prior_pool)
            fresh = _pool_candidates([query], [hits])
            # `_pool_candidates` numbers a one-query batch from zero. Preserve
            # the query's actual position in this scope's full retry history.
            for record in fresh.values():
                for match in record["matches"]:
                    match["sub_query_index"] = query_index
                record["owner_sub_query_index"] = query_index
            for chunk_id, record in fresh.items():
                if chunk_id in prior_pool:
                    record["matches"] = prior_pool[chunk_id]["matches"] + record["matches"]
                    owner = min(
                        record["matches"],
                        key=lambda match: (match["rrf_rank"], match["sub_query_index"]),
                    )
                    record.update(
                        owner_sub_query=owner["sub_query"],
                        owner_sub_query_index=owner["sub_query_index"],
                        owner_rrf_rank=owner["rrf_rank"],
                    )
                prior_pool[chunk_id] = record
            pools[key] = prior_pool
            lost.extend(rt.check_merge_invariant(prior_ids, list(prior_pool)))
            candidates = _select_rerank_candidates(
                prior_pool,
                max_candidates=COMPARISON_RERANK_CANDIDATES_PER_SCOPE,
                max_tokens=COMPARISON_RERANK_TOKENS_PER_SCOPE,
            )
            candidates_by_scope[key] = candidates
            assignments_by_scope[key] = _build_rerank_assignments(
                candidates,
                max_pairs=COMPARISON_RERANK_PAIRS_PER_SCOPE,
                max_tokens=COMPARISON_RERANK_TOKENS_PER_SCOPE,
            )
        update.update(
            comparison_retry_targets=targets,
            comparison_sub_queries=queries_by_scope,
            comparison_retrieved_candidates=pools,
            comparison_candidates=candidates_by_scope,
            comparison_rerank_assignments=assignments_by_scope,
            merge_lost_chunk_ids=lost,
            recovery_seed=state["resolved_question"],
        )
    except Exception as exc:
        update.update(
            correction_error=f"comparison_retrieval:{type(exc).__name__}",
            review_result=_fault_review("recovery_failed"),
        )
    return update


def finalize_answer_node(state: RAGState) -> RAGState:
    review = state.get("review_result") or _fault_review("not_reviewed")
    if review["action"] == "accept" and not state.get("correction_error"):
        return {"final_status": "accepted"}
    # Never return a known-failed draft with just a confidence disclaimer. The
    # partial fallback quotes only passages validated against supplied text.
    excerpts = review.get("supported_excerpts") or []
    answer = "I couldn't verify a complete answer within this turn's correction limit."
    if excerpts:
        answer += "\n\nAvailable source excerpts:\n" + "\n".join(f'- "{e["quote"]}" [{e["chunk_id"]}]' for e in excerpts)
    answer += "\n\nPlease treat this as an incomplete response, not a policy determination."
    status = "review_error" if review.get("reason", "").startswith("review_error") else "limited"
    if review.get("reason") in {"wrong_scope", "wrong_comparison_scope"}:
        status = "scope_error"
    return {"answer": answer, "final_status": status,
            "recovery_exhausted": state.get("retry_attempt", 0) >= state.get("max_retries", 0)}


def _select_final_context(ranked: list, shift: int = 0) -> list[dict]:
    """Choose the k chunks the generator actually sees.

    With no shift this is the plain top-k, unchanged. A reselect shift keeps
    the strongest chunks and swaps the weakest for ones just beyond the cut —
    they are already in the pool and were simply never shown. Fixture G04 is
    exactly that situation: the needed exception sits at rank 11 of a 24-item
    pool while final_k is 10, so no amount of searching harder is required,
    only looking at what was already found.
    """
    if not shift:
        return [c for c, _ in ranked[:FINAL_K]]
    keep = max(1, FINAL_K - shift)
    head = [c for c, _ in ranked[:keep]]
    tail = [c for c, _ in ranked[FINAL_K:FINAL_K + (FINAL_K - keep)]]
    return head + tail


def assess_retrieval_node(state: RAGState) -> RAGState:
    """Deterministic fault check between rerank and generate.

    Finds faults; never certifies sufficiency. See `src/graph/retry.py` for why
    — briefly, fixtures G01/G02 hold identical sufficient evidence at scores
    0.95 and 0.29, so no score function separates sufficient from insufficient.
    A clean pass returns `unknown` and defers the real question to the
    post-generation check, which can see the answer.
    """
    assessment = rt.assess_deterministic(
        state.get("context_chunks") or [],
        state.get("scope"),
        prior_context_ids=None,
    )
    return {
        "assessment": assessment,
        "assessed_action": rt.action_for(assessment),
    }


def _route_after_assess(state: RAGState) -> str:
    action = state.get("assessed_action", "generate")
    if action == "generate":
        return "generate"
    if state.get("retry_attempt", 0) >= state.get("max_retries", 0):
        # Budget spent. Generate from what we have rather than looping; the
        # answer is flagged so an exhausted recovery is visible rather than
        # silently presented as a complete one.
        return "generate"
    return "recover"


def recover_node(state: RAGState) -> RAGState:
    """Run one recovery action, cheapest available first, then re-rank.

    Actions escalate: reselect the pool the generator has not seen (no network),
    search deeper (retrieval cost), reformulate the failing query (LLM cost).
    Retrieval results are merged with the prior pool, never substituted — that
    is the difference between a retry that adds evidence and one that trades
    the evidence it already had, which is the G10 failure.
    """
    attempt = state.get("retry_attempt", 0) + 1
    tried = list(state.get("tried_actions") or [])
    pool = dict(state.get("candidates") or {})
    prior_pool_ids = list(pool.keys())

    action = rt.next_action(attempt, len(pool), FINAL_K, tried)
    if action is None:
        return {"retry_attempt": attempt, "recovery_exhausted": True}

    seed = state.get("recovery_seed") or state.get("resolved_question") or state["question"]
    queries = list(state.get("sub_queries") or [])
    description = ""

    if action == "reselect_context":
        # No search: re-pack the final k from chunks already retrieved but
        # never shown to the generator.
        description = "repacked final k from unseen pool chunks"

    elif action in ("deeper_search", "reformulate"):
        if action == "reformulate":
            queries = [reformulate_query(seed, state["resolved_question"])]
            description = f"reformulated query: {queries[0][:80]}"
        else:
            description = f"re-searched at depth {CANDIDATES_PER_SUB_QUERY * rt.DEPTH_MULTIPLIER}"

        client = qs.connect()
        query_filter = qs.scope_filter(**state["scope"]["filter_args"])
        hit_lists = [
            qs.search_hybrid(
                client,
                embed_query_scoped(
                    q,
                    state["scope"]["filter_args"]["lender_id"],
                    state["scope"]["filter_args"]["document_version"],
                ),
                embed_query_sparse(q),
                limit=CANDIDATES_PER_SUB_QUERY * rt.DEPTH_MULTIPLIER,
                prefetch_limit=CANDIDATES_PER_SUB_QUERY * rt.DEPTH_MULTIPLIER * 2,
                query_filter=query_filter,
            )
            for q in queries
        ]
        fresh = _pool_candidates(queries, hit_lists)
        for chunk_id, record in fresh.items():
            if chunk_id not in pool:
                pool[chunk_id] = record

    lost = rt.check_merge_invariant(prior_pool_ids, list(pool.keys()))
    return {
        "retry_attempt": attempt,
        "tried_actions": [*tried, action],
        "last_action": action,
        "last_action_description": description,
        "candidates": pool,
        "rerank_assignments": _build_rerank_assignments(pool),
        "merge_lost_chunk_ids": lost,
    }


def build_graph(max_retries: int = 0, *, correction_mode: str | None = None,
                review_model: str = JUDGE_MODEL, checkpointer=None, index_selector=None):
    """Compile baseline (default), legacy post-rerank, or post-generation mode.

    Callers must explicitly opt into post_generation. The chat server does;
    existing evaluation runners keep their baseline/legacy configuration.
    max_retries bounds logical correction attempts, not SDK HTTP retries.
    """
    mode = correction_mode or ("post_rerank" if max_retries else "off")
    if mode not in {"off", "post_rerank", "post_generation"} or type(max_retries) is not int or max_retries < 0:
        raise ValueError("invalid correction mode/budget")
    if mode == "post_generation" and max_retries > 1:
        raise ValueError("the first post-generation version allows at most one recovery")
    graph = StateGraph(RAGState)
    def add(name, function):
        wrapped = observed_node(name, guard_node(name, function))
        def pinned(state):
            from src.store.index_context import bind
            from src.store.index_builds import validate_binding
            binding = state.get("index_binding") or {}
            # Begin-turn chooses afresh. Every subsequent node, including a
            # resumed/retried node, uses the checkpoint's concrete binding.
            if binding and name != "begin_turn":
                validate_binding(binding)
            with bind(binding if name != "begin_turn" else None):
                return wrapped(state)
        graph.add_node(name, pinned)
    def begin(state):
        reset = begin_turn_node(state, mode, max_retries, review_model)
        reset["index_binding"] = index_selector() if index_selector else {}
        return reset
    add("begin_turn", begin)
    add("contextualize", contextualize_node)
    add("catalog_lenders", catalog_lenders_node)
    add("conversation_history", conversation_history_node)
    add("resolve_scope", resolve_scope_node)
    add("resolve_comparison_scopes", resolve_comparison_scopes_node)
    add("clarify", clarify_node)
    add("unresolved", unresolved_node)
    add("decline", decline_node)
    add("reuse_evidence", reuse_evidence_node)
    add("decompose", decompose_node)
    add("retrieve", retrieve_node)
    add("rerank", rerank_node)
    add("generate", generate_node)
    add("plan_comparison", plan_comparison_node)
    add("retrieve_comparison", retrieve_comparison_node)
    add("rerank_comparison", rerank_comparison_node)
    add("generate_comparison", generate_comparison_node)

    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "contextualize")
    graph.add_conditional_edges("contextualize", _route_after_contextualize,
                                {"catalog_lenders": "catalog_lenders",
                                 "conversation_history": "conversation_history",
                                 "resolve_comparison_scopes": "resolve_comparison_scopes",
                                 "resolve_scope": "resolve_scope"})
    graph.add_edge("catalog_lenders", END)
    graph.add_edge("conversation_history", END)
    graph.add_conditional_edges(
        "resolve_scope",
        _route_after_scope,
        {
            "decompose": "decompose",
            "clarify": "clarify",
            "unresolved": "unresolved",
            "decline": "decline",
            "reuse_evidence": "reuse_evidence",
        },
    )
    graph.add_conditional_edges(
        "resolve_comparison_scopes",
        _route_after_comparison_scope,
        {
            "plan_comparison": "plan_comparison",
            "clarify": "clarify",
            "unresolved": "unresolved",
        },
    )
    # Loop back rather than resolving scope from the reply directly: the
    # resolver already validates against the manifest, rejects versions that do
    # not exist, and handles dates. A second path would be a weaker duplicate
    # that can disagree with it.
    graph.add_conditional_edges(
        "clarify",
        _route_after_clarify,
        {
            "resolve_scope": "resolve_scope",
            "resolve_comparison_scopes": "resolve_comparison_scopes",
        },
    )
    graph.add_edge("unresolved", END)
    graph.add_edge("decline", END)
    graph.add_edge("reuse_evidence", "review_answer" if mode == "post_generation" else END)
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("plan_comparison", "retrieve_comparison")
    graph.add_edge("retrieve_comparison", "rerank_comparison")

    if mode == "post_generation":
        add("check_evidence", check_evidence_node)
        add("review_answer", review_answer_node)
        add("revise_answer", revise_answer_node)
        add("targeted_retrieve", targeted_retrieve_node)
        add("check_comparison_evidence", check_comparison_evidence_node)
        add("review_comparison", review_comparison_node)
        add("revise_comparison", revise_comparison_node)
        add("targeted_comparison_retrieve", targeted_comparison_retrieve_node)
        add("finalize", finalize_answer_node)
        graph.add_edge("rerank", "check_evidence")
        graph.add_conditional_edges("check_evidence", _after_evidence,
                                    {"generate": "generate", "targeted_retrieve": "targeted_retrieve", "finalize": "finalize"})
        graph.add_edge("generate", "review_answer")
        graph.add_conditional_edges("review_answer", _correction_route,
                                    {"revise_answer": "revise_answer", "targeted_retrieve": "targeted_retrieve", "clarify": "clarify", "finalize": "finalize"})
        graph.add_edge("revise_answer", "review_answer")
        graph.add_conditional_edges("targeted_retrieve", lambda state: "finalize" if state.get("correction_error") else "rerank",
                                    {"finalize": "finalize", "rerank": "rerank"})
        graph.add_edge("rerank_comparison", "check_comparison_evidence")
        graph.add_conditional_edges(
            "check_comparison_evidence",
            _after_comparison_evidence,
            {
                "generate_comparison": "generate_comparison",
                "targeted_comparison_retrieve": "targeted_comparison_retrieve",
                "finalize": "finalize",
            },
        )
        graph.add_edge("generate_comparison", "review_comparison")
        graph.add_conditional_edges(
            "review_comparison",
            _comparison_correction_route,
            {
                "revise_comparison": "revise_comparison",
                "targeted_comparison_retrieve": "targeted_comparison_retrieve",
                "clarify": "clarify",
                "finalize": "finalize",
            },
        )
        graph.add_edge("revise_comparison", "review_comparison")
        graph.add_conditional_edges(
            "targeted_comparison_retrieve",
            lambda state: "finalize" if state.get("correction_error") else "rerank_comparison",
            {"finalize": "finalize", "rerank_comparison": "rerank_comparison"},
        )
        graph.add_edge("finalize", END)
    elif mode == "off":
        # Baseline route: no reviewer or recovery nodes.
        graph.add_edge("rerank", "generate")
    else:
        add("assess_retrieval", assess_retrieval_node)
        add("recover", recover_node)
        graph.add_edge("rerank", "assess_retrieval")
        graph.add_conditional_edges(
            "assess_retrieval",
            _route_after_assess,
            {"generate": "generate", "recover": "recover"},
        )
        # Recovery re-enters the reranker rather than the retriever: the
        # recovery action has already put whatever it found into the pool, and
        # the pool has to be re-scored as a whole for the merge to mean
        # anything. Looping to `retrieve` would re-run the original searches.
        graph.add_edge("recover", "rerank")

    if mode != "post_generation":
        graph.add_edge("generate", END)
        graph.add_edge("rerank_comparison", "generate_comparison")
        graph.add_edge("generate_comparison", END)
    # The checkpointer is what makes `interrupt` resumable: state is persisted
    # per `thread_id`, so a resumed run continues from the paused node instead
    # of restarting. Offline callers default to MemorySaver; the chat service
    # supplies PostgresSaver so conversation state survives service restarts.
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())
