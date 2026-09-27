"""One turn through the compiled graph — shared by the chat server and the
conversation-eval runner.

This existed only inside `src/app/server.py` until the eval runner needed it
too. Duplicating it would let the two callers silently diverge — the eval
could end up scoring a pipeline that isn't quite the one users actually talk
to. There is exactly one way a turn is executed; both callers use it.
"""

from __future__ import annotations

import uuid

from langgraph.types import Command
from src.guardrails.tracing import traceable
from src.store.state_store import evidence_hash
import time
from src.util.telemetry import summarize
from src.guardrails.core import GuardrailError, blocked_result, capture_guards
from src.guardrails.checks import request_check, final_check
from src.guardrails.privacy import sanitize, trace_payload
from src.util.telemetry import capture_request
from langsmith import Client, tracing_context
from langsmith.utils import tracing_is_enabled
from functools import lru_cache


@lru_cache(maxsize=1)
def _trace_client():
    return Client(hide_inputs=trace_payload, hide_outputs=trace_payload,
                  hide_metadata=trace_payload)


def _guard_diagnostics(state: dict) -> dict:
    """Return non-content diagnostics for an internally blocked turn.

    Production callers must not receive the rejected draft or source text.  An
    offline evaluator still needs to distinguish "retrieval never happened"
    from "retrieval succeeded and the final guard rejected the draft".  Keep
    only routing, provenance metadata, node telemetry and citation tokens.
    """
    from src.graph.answer_review import BRACKET_TOKEN_RE

    def chunk_ref(chunk: dict) -> dict:
        return {
            key: chunk.get(key)
            for key in (
                "chunk_id", "lender_id", "document_id", "document_version",
            )
        }

    answer = state.get("answer") or ""
    citation_tokens = [
        token.strip()
        for token in BRACKET_TOKEN_RE.findall(answer)
        if "::" in token or token.strip().isdigit()
    ]
    return {
        "behavior": state.get("behavior"),
        "scope": state.get("scope") or {},
        "comparison_scopes": state.get("comparison_scopes") or [],
        "comparison_context": {
            key: [chunk_ref(chunk) for chunk in chunks]
            for key, chunks in (state.get("comparison_context") or {}).items()
        },
        "comparison_retry_targets": state.get("comparison_retry_targets") or [],
        "merge_lost_chunk_ids": state.get("merge_lost_chunk_ids") or [],
        "node_events": state.get("node_events") or [],
        "usage_summary": state.get("usage_summary") or {},
        "retry_attempt": state.get("retry_attempt", 0),
        "recovery_exhausted": bool(state.get("recovery_exhausted", False)),
        "citation_tokens": citation_tokens,
    }


def run_turn(
    graph,
    thread_id: str,
    message: str,
    *,
    include_guard_diagnostics: bool = False,
) -> dict:
    """Untraced entry boundary shared by HTTP and evaluation/CLI callers.

    ``include_guard_diagnostics`` is for controlled offline evaluation only.
    It never includes the rejected answer or evidence text.
    """
    started = time.perf_counter()
    with capture_guards() as events, capture_request() as request_usage:
        previous = None
        try:
            message = request_check(message)
            previous = sanitize(graph.get_state(_config(thread_id)).values, "prior_memory")
            with tracing_context(client=_trace_client() if tracing_is_enabled() else None):
                result = _run_turn(graph, thread_id, message)
        except GuardrailError as exc:
            failed_diagnostics = None
            if include_guard_diagnostics and previous is not None:
                failed_state = sanitize(
                    graph.get_state(_config(thread_id)).values,
                    "guard_diagnostics",
                )
                failed_diagnostics = _guard_diagnostics(failed_state)
            if previous is not None:
                # Do not let a rejected turn change the scope/history used by
                # the next follow-up. Input rejections never resume an interrupt.
                graph.update_state(_config(thread_id), {
                    "history": previous.get("history", []),
                    "scope": previous.get("scope") or {},
                    "prior_scope": previous.get("prior_scope"),
                    "answer": "", "context_chunks": [], "candidates": {},
                    "retrieved_candidates": {}, "reranked": [], "rerank_assignments": [],
                    "comparison_scopes": [], "comparison_context": {},
                    "comparison_candidates": {}, "comparison_retrieved_candidates": {},
                })
            result = blocked_result(exc.code)
            result["awaiting_clarification"] = _is_paused(graph, thread_id)
            if failed_diagnostics is not None:
                result["guard_diagnostics"] = failed_diagnostics
        result["guard_events"] = list(events)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        result["request_usage"] = summarize([{"node": "request", "events": request_usage, "elapsed_ms": elapsed}])
        guard_calls = [e for e in request_usage if e.get("operation") == "guardrail_screen"]
        result["guard_usage"] = summarize([{"node": "guards", "events": guard_calls, "elapsed_ms": 0}])
        result["request_usage"]["request_elapsed_ms"] = elapsed
        return result


def _is_paused(graph, thread_id: str) -> bool:
    """Is this thread waiting on a clarification reply?

    Checking `.next` alone is wrong: it is also non-empty for a thread that
    has been `update_state`-seeded but never invoked — `.next` there points at
    the entry node, which has not run yet, not at a paused one. The only
    reliable signal is a real pending `Interrupt` on one of the thread's tasks.
    Production never hit this, since `/chat` only ever seeds `history` after a
    real `invoke` completes; the eval runner's `seed_thread` broke that
    assumption by writing state before the first turn ever runs.
    """
    return any(t.interrupts for t in graph.get_state(_config(thread_id)).tasks)


def _resolved(state: dict) -> bool:
    return bool((state.get("scope") or {}).get("status") == "resolved")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@traceable(run_type="chain", name="rag_turn")
def _run_turn(graph, thread_id: str, message: str) -> dict:
    """Execute one user message on `thread_id` and return the graph's final state.

    Branches on whether the thread is paused, exactly as `/chat` does: a
    pending interrupt means `message` is a clarification reply, resumed with
    `Command`; otherwise it is a new question, and the per-turn fields are
    reset explicitly while `history`/`prior_scope` are carried forward from
    whatever is already checkpointed on this thread.

    A completed (non-interrupted) turn is appended to `history` before
    returning, so the next call on this thread sees it — same as `/chat`.
    """
    config = _config(thread_id)
    started = time.perf_counter()
    paused = _is_paused(graph, thread_id)
    snapshot = sanitize(graph.get_state(config).values, "prior_memory")
    previous_nodes = snapshot.get("node_events", []) if paused else []

    if paused:
        # A pre-guardrails paused thread may contain raw legacy input. Do not
        # resume it into model calls until its stored state is clean.
        if snapshot != graph.get_state(config).values:
            raise GuardrailError("legacy_memory_requires_cleanup")
        final = graph.invoke(Command(resume=message), config=config)
    else:
        previous = sanitize(graph.get_state(config).values, "prior_memory")
        history = list(previous.get("history") or [])
        prior_scope = previous.get("scope") if _resolved(previous) else None

        final = graph.invoke(
            {
                "turn_id": str(uuid.uuid4()),
                "question": message,
                "resolved_question": message,
                "clarification_rounds": 0,
                "clarifications": [],
                "history": history,
                "prior_scope": prior_scope,
                "scope_inherited": False,
            },
            config=config,
        )

    nodes = final.get("node_events", [])
    final["usage_summary"] = summarize(nodes)
    # Human think time is not included. A resume reports this invocation's wall
    # time separately from cumulative completed-node time for the logical turn.
    final["usage_summary"]["invocation_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    final["usage_summary"]["prior_completed_nodes"] = len(previous_nodes)
    if final.get("__interrupt__"):
        # Interrupt values can contain a generated clarification question.
        for item in final["__interrupt__"]:
            if sanitize(item.value, "clarification_output") != item.value:
                raise GuardrailError("unsafe_clarification")
        return final

    final = final_check(final)
    scope = final.get("scope") or {}
    history_scope = {key: scope[key] for key in (
        "status", "lender_id", "lender_name", "document_id", "document_version"
    ) if key in scope}
    comparison_scopes = [
        {key: scope[key] for key in (
            "status", "lender_id", "lender_name", "document_id", "document_version"
        ) if key in scope}
        for scope in (final.get("comparison_scopes") or [])
    ]
    comparison_chunk_ids = {
        key: [chunk["chunk_id"] for chunk in chunks]
        for key, chunks in (final.get("comparison_context") or {}).items()
    }
    graph.update_state(
        config,
        {
            "usage_summary": final["usage_summary"],
            "history": [
                *(final.get("history") or []),
                {
                    "question": final.get("resolved_question") or final["question"],
                    "answer": final.get("answer") or "",
                    "index_binding": final.get("index_binding") or {},
                    # Ids, not full chunk dicts: cheap to store, and the reuse
                    # path re-hydrates text by id from Qdrant rather than
                    # carrying a growing pile of chunk text in every checkpoint.
                    "chunk_ids": [c["chunk_id"] for c in (final.get("context_chunks") or [])],
                    # Lets a later reuse_evidence_node detect whether the ids
                    # above still point at the same chunks -- see _evidence_fault.
                    "evidence_hash": evidence_hash(
                        c["chunk_id"] for c in (final.get("context_chunks") or [])
                    ),
                    # These fields make conversation questions answerable from
                    # their own provenance. Older checkpoints lack them and
                    # are handled conservatively using saved chunk IDs.
                    "scope": history_scope,
                    "scopes": comparison_scopes,
                    "chunk_ids_by_scope": comparison_chunk_ids,
                    "behavior": final.get("behavior") or "",
                    "turn_type": (
                        "policy_comparison"
                        if comparison_scopes and final.get("context_chunks")
                        and final.get("final_status") in {"accepted", "not_reviewed"}
                        else
                        "policy_answer"
                        if (final.get("scope") or {}).get("status") == "resolved"
                        and final.get("context_chunks")
                        and final.get("final_status") in {"accepted", "not_reviewed"}
                        else final.get("behavior") or "other"
                    ),
                    "answer_basis": final.get("answer_basis") or {},
                },
            ]
        },
    )
    return final


def seed_thread(graph, thread_id: str, history: list[dict], scope: dict | None) -> None:
    """Pre-load a thread's checkpoint with fixture history, before any turn runs.

    For the eval runner only — a real conversation always builds `history`
    turn by turn through `run_turn`. This writes the same shape directly, so
    the first scored step on this thread sees prior context exactly as it
    would after genuinely having that conversation.
    """
    update: dict = {"history": history}
    if scope is not None:
        update["scope"] = scope
    graph.update_state(_config(thread_id), update)
