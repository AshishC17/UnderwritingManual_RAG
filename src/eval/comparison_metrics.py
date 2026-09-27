"""Mechanical metrics for cross-document comparison runs.

These metrics do not ask an LLM whether prose is correct. They measure whether
the graph chose the comparison route, locked both authorities, retrieved each
required evidence group under its own scope, kept the evidence partitions
clean, targeted the intended retry side, and cited supplied chunks. Generation
claim recall, groundedness, forbidden claims and relevance remain judge-scored
and are reported separately by the comparison eval runner.
"""
from __future__ import annotations

from src.eval.generation_metrics import CITATION_RE
from src.resolve.scope import scope_key


def _scope_tuple(scope: dict) -> tuple[str | None, str | None, str | None]:
    return (
        scope.get("lender_id"),
        scope.get("document_id"),
        scope.get("document_version"),
    )


def score_comparison_execution(case: dict, final: dict) -> dict:
    expected = case["expected"]
    interrupted = bool(final.get("__interrupt__"))
    expected_scopes = expected.get("scopes") or []
    actual_scopes = final.get("comparison_scopes") or []
    expected_scope_set = {_scope_tuple(scope) for scope in expected_scopes}
    actual_scope_set = {_scope_tuple(scope) for scope in actual_scopes}
    predicted_behavior = "clarify" if interrupted else final.get("behavior")

    contexts = final.get("comparison_context") or {}
    actual_ids = {
        key: {chunk["chunk_id"] for chunk in chunks}
        for key, chunks in contexts.items()
    }
    groups_by_scope: dict[str, list[dict]] = {}
    for group in expected.get("evidence_groups") or []:
        groups_by_scope.setdefault(group["scope_key"], []).append(group)

    recall_by_scope: dict[str, float | None] = {}
    groups_found = 0
    groups_total = 0
    for key, groups in groups_by_scope.items():
        found = sum(bool(actual_ids.get(key, set()) & set(group["any_of_chunk_ids"]))
                    for group in groups)
        recall_by_scope[key] = found / len(groups) if groups else None
        groups_found += found
        groups_total += len(groups)

    wrong_scope_chunks: list[str] = []
    total_chunks = 0
    expected_by_key = {scope_key(scope): scope for scope in expected_scopes}
    for key, chunks in contexts.items():
        total_chunks += len(chunks)
        authority = expected_by_key.get(key)
        for chunk in chunks:
            if authority is None or _scope_tuple(chunk) != _scope_tuple(authority):
                wrong_scope_chunks.append(chunk.get("chunk_id", "<missing-id>"))

    supplied_ids = {
        chunk["chunk_id"] for chunks in contexts.values() for chunk in chunks
    }
    diagnostic_citations = final.get("citation_tokens")
    cited = (
        list(diagnostic_citations)
        if isinstance(diagnostic_citations, list)
        else CITATION_RE.findall(final.get("answer") or "")
    )
    valid_cited = [chunk_id for chunk_id in cited if chunk_id in supplied_ids]
    cited_scopes = {
        key for key, ids in actual_ids.items() if set(valid_cited) & ids
    }

    expected_retry = expected.get("retry_scope_keys") or []
    actual_retry = final.get("comparison_retry_targets") or []
    retry_side_accuracy = (
        int(set(actual_retry) == set(expected_retry)) if expected_retry else None
    )
    expected_answer = expected["outcome"] == "answer"
    guard_blocked = bool(final.get("guard_reason"))
    # A graph can finish normally yet deliberately withhold the draft after a
    # reviewer/correction failure.  That is a safe operational outcome, but it
    # is not a successful answer.  Older fixtures/runs may not carry this field,
    # so absence retains the previous backwards-compatible interpretation.
    final_status = final.get("final_status")
    answer_released = final_status in (None, "accepted", "not_reviewed")
    outcome_pass = (
        bool(not interrupted and not guard_blocked and final.get("behavior")
             and answer_released)
        if expected_answer else interrupted
    )
    scope_pass = (
        actual_scope_set == expected_scope_set
        if expected_scopes else not actual_scopes
    )
    bilateral = (
        int(bool(recall_by_scope) and len(recall_by_scope) == 2
            and all(value == 1.0 for value in recall_by_scope.values()))
        if groups_total else None
    )
    behavior_pass = predicted_behavior == expected["behavior"]
    isolation_pass = not wrong_scope_chunks
    orchestration_pass = bool(
        behavior_pass and outcome_pass and scope_pass and isolation_pass
        and (retry_side_accuracy in (None, 1))
    )

    usage = final.get("usage_summary") or {}
    events = [
        event
        for node in (final.get("node_events") or [])
        for event in node.get("events", [])
        if event.get("kind") == "api_call"
    ]
    return {
        "predicted_behavior": predicted_behavior,
        "final_status": final_status,
        "behavior_pass": behavior_pass,
        "outcome_pass": outcome_pass,
        "both_scope_accuracy": int(scope_pass),
        "evidence_group_recall": groups_found / groups_total if groups_total else None,
        "evidence_group_recall_by_scope": recall_by_scope,
        "bilateral_full_coverage": bilateral,
        "context_chunks_by_scope": {key: len(chunks) for key, chunks in contexts.items()},
        "wrong_scope_chunk_rate": (
            len(wrong_scope_chunks) / total_chunks if total_chunks else 0.0
        ),
        "wrong_scope_chunk_ids": wrong_scope_chunks,
        "citation_validity": len(valid_cited) / len(cited) if cited else None,
        "citation_scope_coverage": (
            len(cited_scopes & set(expected_by_key)) / len(expected_by_key)
            if expected_by_key else None
        ),
        "retry_side_accuracy": retry_side_accuracy,
        "retry_targets": actual_retry,
        "merge_preservation_pass": not bool(final.get("merge_lost_chunk_ids")),
        "orchestration_pass": orchestration_pass,
        "latency_ms": usage.get("request_elapsed_ms") or usage.get("invocation_elapsed_ms"),
        "provider_calls": usage.get("api_calls"),
        "input_tokens": (
            sum(event["input_tokens"] for event in events)
            if events and all(event.get("input_tokens") is not None for event in events)
            else None
        ),
        "output_tokens": (
            sum(event["output_tokens"] for event in events)
            if events and all(event.get("output_tokens") is not None for event in events)
            else None
        ),
    }


def aggregate_comparison(records: list[dict]) -> dict:
    def mean(field: str):
        values = [record[field] for record in records if record.get(field) is not None]
        return sum(values) / len(values) if values else None

    return {
        "cases": len(records),
        "behavior_accuracy": mean("behavior_pass"),
        "outcome_accuracy": mean("outcome_pass"),
        "both_scope_accuracy": mean("both_scope_accuracy"),
        "evidence_group_recall": mean("evidence_group_recall"),
        "bilateral_full_coverage_rate": mean("bilateral_full_coverage"),
        "wrong_scope_chunk_rate": mean("wrong_scope_chunk_rate"),
        "citation_validity": mean("citation_validity"),
        "citation_scope_coverage": mean("citation_scope_coverage"),
        "targeted_retry_accuracy": mean("retry_side_accuracy"),
        "merge_preservation_rate": mean("merge_preservation_pass"),
        "orchestration_success_rate": mean("orchestration_pass"),
        "mean_latency_ms": mean("latency_ms"),
        "mean_provider_calls": mean("provider_calls"),
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "denominators": {
            field: sum(record.get(field) is not None for record in records)
            for field in (
                "evidence_group_recall", "bilateral_full_coverage",
                "citation_validity", "citation_scope_coverage",
                "retry_side_accuracy", "latency_ms", "provider_calls",
                "input_tokens", "output_tokens",
            )
        },
    }
