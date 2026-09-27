from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.graph.pipeline import (
    build_graph,
    resolve_comparison_scopes_node,
    retrieve_comparison_node,
    targeted_comparison_retrieve_node,
)
from src.graph.answer_review import (
    COMPARISON_PROMPT,
    review_comparison_draft,
    validate_comparison_review,
)
from src.graph.runtime import run_turn, seed_thread
from src.guardrails.core import GuardrailError
from src.guardrails.checks import comparison_evidence_check
from src.ingest.manifest import load_manifest
from src.resolve.scope import resolve_comparison_scopes, scope_key


def _scope(lender: str, version: str = "V1") -> dict:
    if lender == "novacred":
        return {
            "status": "resolved", "lender_id": lender, "lender_name": "NovaCred Financial",
            "document_id": "novacred_flex_uw", "document_version": version,
            "filter_args": {"lender_id": lender, "document_id": "novacred_flex_uw",
                            "document_version": version},
        }
    return {
        "status": "resolved", "lender_id": lender, "lender_name": "LumenTrail Financial",
        "document_id": "lumentrail_pathway_uw", "document_version": version,
        "filter_args": {"lender_id": lender, "document_id": "lumentrail_pathway_uw",
                        "document_version": version},
    }


def _chunks(scope: dict, count: int = 6) -> list[dict]:
    prefix = "NovaCred_UW" if scope["lender_id"] == "novacred" else "LumenTrail_UW"
    return [{
        "chunk_id": f"{prefix}_{scope['document_version']}_pdf.pdf::{index:04d}",
        "text": f"Policy evidence {index} for {scope['lender_name']}.",
        "lender_id": scope["lender_id"],
        "document_id": scope["document_id"],
        "document_version": scope["document_version"],
        "section": "Policy",
        "token_count": 20,
    } for index in range(count)]


class _Filter:
    def __init__(self, lender_id: str, document_version: str = "V1"):
        self.lender_id = lender_id
        self.document_version = document_version

    def model_dump(self, **_kwargs):
        return {"must": [{"key": "lender_id", "match": {"value": self.lender_id}}]}


def _filter(**kwargs):
    return _Filter(kwargs["lender_id"], kwargs.get("document_version", "V1"))


class ComparisonScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = load_manifest("config/corpus_manifest.json", require_files=False)

    def test_standalone_comparison_defaults_each_lender_to_current(self):
        result = resolve_comparison_scopes(
            "Compare the current NovaCred and LumenTrail credit-freeze policies.",
            specs=self.specs,
        )
        self.assertEqual("resolved", result["status"])
        self.assertEqual(
            [("novacred", "V1"), ("lumentrail", "V1")],
            [(scope["lender_id"], scope["document_version"]) for scope in result["scopes"]],
        )

    def test_explicit_versions_are_attached_to_their_lenders(self):
        result = resolve_comparison_scopes(
            "Compare NovaCred V0 versus LumenTrail V1 on Code 128.", specs=self.specs
        )
        self.assertEqual(
            [("novacred", "V0"), ("lumentrail", "V1")],
            [(scope["lender_id"], scope["document_version"]) for scope in result["scopes"]],
        )

    def test_followup_preserves_prior_side_version_only(self):
        prior = _scope("lumentrail", "V0")
        result = resolve_comparison_scopes(
            "Compare that with NovaCred.", prior_scope=prior, specs=self.specs
        )
        self.assertEqual(
            [("lumentrail", "V0"), ("novacred", "V1")],
            [(scope["lender_id"], scope["document_version"]) for scope in result["scopes"]],
        )

    def test_current_word_overrides_historical_prior_side(self):
        prior = _scope("lumentrail", "V0")
        result = resolve_comparison_scopes(
            "Compare the current LumenTrail and NovaCred policies.",
            prior_scope=prior,
            specs=self.specs,
        )
        self.assertEqual(["V1", "V1"], [scope["document_version"] for scope in result["scopes"]])

    def test_missing_antecedent_fails_closed(self):
        result = resolve_comparison_scopes(
            "Compare that with NovaCred.", specs=self.specs
        )
        self.assertEqual("clarification", result["status"])
        self.assertEqual("missing_comparison_target", result["reason"])

    def test_resolver_materializes_two_filters_without_overloading_scope(self):
        state = {
            "question": "Compare current NovaCred and LumenTrail policies.",
            "resolved_question": "Compare current NovaCred and LumenTrail policies.",
            "prior_scope": None,
        }
        with patch("src.graph.pipeline.qs.scope_filter", side_effect=_filter):
            result = resolve_comparison_scopes_node(state)
        self.assertEqual({}, result["scope"])
        self.assertEqual(2, len(result["comparison_scopes"]))
        self.assertEqual(set(result["comparison_filters"]), {
            scope_key(scope) for scope in result["comparison_scopes"]
        })


class ComparisonIsolationTests(unittest.TestCase):
    def test_runtime_review_rubric_covers_consequences_and_acronym_grounding(self):
        self.assertIn("operational meaning or consequence", COMPARISON_PROMPT)
        self.assertIn("acronym expansion", COMPARISON_PROMPT)

    def test_runtime_reviewer_receives_exact_composite_scope_keys(self):
        scopes = [_scope("novacred"), _scope("lumentrail")]
        contexts = {scope_key(scope): _chunks(scope, 1) for scope in scopes}
        keys = [scope_key(scope) for scope in scopes]
        data = {
            "requirements": [
                {
                    "need": "NovaCred rule", "scope_key": keys[0],
                    "evidence": "present", "answer": "covered",
                    "source_ids": [contexts[keys[0]][0]["chunk_id"]],
                },
                {
                    "need": "LumenTrail rule", "scope_key": keys[1],
                    "evidence": "present", "answer": "covered",
                    "source_ids": [contexts[keys[1]][0]["chunk_id"]],
                },
            ],
            "unsupported_claims": [], "citation_issues": [],
            "needs_clarification": False, "clarification_question": "",
            "reason": "complete", "supported_excerpts": [],
        }
        state = {
            "question": "Compare the rules.",
            "resolved_question": "Compare the rules.",
            "history": [], "clarifications": [],
            "comparison_scopes": scopes, "comparison_context": contexts,
            "answer": "Comparison draft.",
        }
        with patch("src.graph.answer_review._ask", return_value=data) as ask:
            result = review_comparison_draft(state)

        prompt = ask.call_args.args[0]
        self.assertIn(f'"scope_key": "{keys[0]}"', prompt)
        self.assertIn(f'"scope_key": "{keys[1]}"', prompt)
        self.assertEqual("accept", result["action"])

    def test_retrieval_runs_one_filtered_search_per_scope(self):
        scopes = [_scope("novacred"), _scope("lumentrail")]
        keys = [scope_key(scope) for scope in scopes]
        state = {
            "comparison_scopes": scopes,
            "comparison_sub_queries": {key: ["credit freeze"] for key in keys},
        }
        chunks = {scope["lender_id"]: _chunks(scope, 2) for scope in scopes}

        def search(*_args, **kwargs):
            lender = kwargs["query_filter"].lender_id
            return [SimpleNamespace(payload=chunk, score=1.0 / (index + 1))
                    for index, chunk in enumerate(chunks[lender])]

        with (
            patch("src.graph.pipeline.qs.connect", return_value=object()),
            patch("src.graph.pipeline.qs.scope_filter", side_effect=_filter),
            patch("src.graph.pipeline.embed_query_scoped", return_value=[0.0]),
            patch("src.graph.pipeline.embed_query_sparse", return_value=object()),
            patch("src.graph.pipeline.qs.search_hybrid", side_effect=search) as hybrid,
        ):
            result = retrieve_comparison_node(state)

        self.assertEqual(2, hybrid.call_count)
        self.assertEqual(set(keys), set(result["comparison_retrieved_candidates"]))
        for scope in scopes:
            key = scope_key(scope)
            self.assertTrue(all(
                record["chunk"]["lender_id"] == scope["lender_id"]
                for record in result["comparison_retrieved_candidates"][key].values()
            ))

    def test_guard_rejects_chunk_borrowed_from_other_scope(self):
        scopes = [_scope("novacred"), _scope("lumentrail")]
        contexts = {
            scope_key(scopes[0]): [_chunks(scopes[1], 1)[0]],
            scope_key(scopes[1]): _chunks(scopes[1], 1),
        }
        with patch("src.guardrails.checks.screen"):
            with self.assertRaises(GuardrailError) as raised:
                comparison_evidence_check(contexts, scopes)
        self.assertEqual("wrong_scope", raised.exception.code)

    def test_review_requires_both_scopes_for_comparison_claim(self):
        scopes = [_scope("novacred"), _scope("lumentrail")]
        contexts = {scope_key(scope): _chunks(scope, 1) for scope in scopes}
        data = {
            "requirements": [{
                "need": "Compare the two rules",
                "scope_key": "comparison",
                "evidence": "present",
                "answer": "covered",
                "source_ids": [contexts[scope_key(scopes[0])][0]["chunk_id"]],
            }],
            "unsupported_claims": [], "citation_issues": [],
            "needs_clarification": False, "clarification_question": "",
            "reason": "complete", "supported_excerpts": [],
        }
        with self.assertRaisesRegex(ValueError, "must cite both scopes"):
            validate_comparison_review(data, scopes, contexts)

    def test_review_records_the_scope_that_needs_retry(self):
        scopes = [_scope("novacred"), _scope("lumentrail")]
        contexts = {scope_key(scope): _chunks(scope, 1) for scope in scopes}
        missing_key = scope_key(scopes[1])
        data = {
            "requirements": [{
                "need": "Find the LumenTrail rule",
                "scope_key": missing_key,
                "evidence": "missing", "answer": "missing", "source_ids": [],
            }],
            "unsupported_claims": [], "citation_issues": [],
            "needs_clarification": False, "clarification_question": "",
            "reason": "one side incomplete", "supported_excerpts": [],
        }
        result = validate_comparison_review(data, scopes, contexts)
        self.assertEqual("retrieve", result["action"])
        self.assertEqual([missing_key], result["missing_scope_keys"])

    def test_retry_searches_only_missing_scope_and_preserves_other_pool(self):
        scopes = [_scope("novacred"), _scope("lumentrail")]
        keys = [scope_key(scope) for scope in scopes]
        initial = {
            "comparison_scopes": scopes,
            "comparison_sub_queries": {key: ["credit freeze"] for key in keys},
            "comparison_retrieved_candidates": {},
            "comparison_candidates": {},
            "comparison_rerank_assignments": {},
            "review_result": {
                "missing_scope_keys": [keys[1]],
                "requirements": [{
                    "need": "LumenTrail consequence",
                    "scope_key": keys[1], "evidence": "missing",
                }],
            },
            "resolved_question": "Compare credit-freeze handling.",
            "retry_attempt": 0, "tried_actions": [],
        }
        for scope, key in zip(scopes, keys):
            chunk = _chunks(scope, 1)[0]
            initial["comparison_retrieved_candidates"][key] = {
                chunk["chunk_id"]: {
                    "chunk": chunk,
                    "matches": [{"sub_query": "credit freeze", "sub_query_index": 0,
                                 "rrf_rank": 1, "rrf_score": 1.0}],
                    "owner_sub_query": "credit freeze",
                    "owner_sub_query_index": 0, "owner_rrf_rank": 1,
                }
            }

        fresh_chunk = _chunks(scopes[1], 2)[1]
        with (
            patch("src.graph.pipeline.qs.connect", return_value=object()),
            patch("src.graph.pipeline.qs.scope_filter", side_effect=_filter),
            patch("src.graph.pipeline.embed_query_scoped", return_value=[0.0]),
            patch("src.graph.pipeline.embed_query_sparse", return_value=object()),
            patch("src.graph.pipeline.qs.search_hybrid", return_value=[
                SimpleNamespace(payload=fresh_chunk, score=1.0)
            ]) as hybrid,
        ):
            result = targeted_comparison_retrieve_node(initial)

        self.assertEqual(1, hybrid.call_count)
        self.assertEqual("lumentrail", hybrid.call_args.kwargs["query_filter"].lender_id)
        self.assertEqual(
            set(initial["comparison_retrieved_candidates"][keys[0]]),
            set(result["comparison_retrieved_candidates"][keys[0]]),
        )
        self.assertEqual(2, len(result["comparison_sub_queries"][keys[1]]))
        self.assertEqual(
            1,
            result["comparison_retrieved_candidates"][keys[1]][fresh_chunk["chunk_id"]]
            ["matches"][0]["sub_query_index"],
        )


class ComparisonGraphTests(unittest.TestCase):
    def _run(self, question: str, *, history=None, prior_scope=None, classifier=None):
        def search(*_args, **kwargs):
            query_filter = kwargs["query_filter"]
            scoped = _scope(query_filter.lender_id, query_filter.document_version)
            chunks = _chunks(scoped)
            return [SimpleNamespace(payload=chunk, score=1.0 / (index + 1))
                    for index, chunk in enumerate(chunks)]

        def score(_query, chunks, **_kwargs):
            return [(chunk, 1.0 - index / 100) for index, chunk in enumerate(chunks)]

        def comparison_answer(_question, scopes, contexts):
            citations = [contexts[scope_key(scope)][0]["chunk_id"] for scope in scopes]
            return f"First policy [{citations[0]}]. Second policy [{citations[1]}]."
        graph = build_graph(correction_mode="off")
        thread = "comparison-test"
        if history is not None:
            seed_thread(graph, thread, history, prior_scope)

        patches = [
            patch("src.guardrails.checks.screen"),
            patch("src.graph.pipeline.decompose_query", side_effect=lambda q: [q]),
            patch("src.graph.pipeline.qs.connect", return_value=object()),
            patch("src.graph.pipeline.qs.scope_filter", side_effect=_filter),
            patch("src.graph.pipeline.embed_query_scoped", return_value=[0.0]),
            patch("src.graph.pipeline.embed_query_sparse", return_value=object()),
            patch("src.graph.pipeline.qs.search_hybrid", side_effect=search),
            patch("src.graph.pipeline.rerank", side_effect=score),
            patch("src.graph.pipeline.generate_comparison", side_effect=comparison_answer),
            patch("src.graph.runtime.tracing_is_enabled", return_value=False),
        ]
        if classifier is not None:
            patches.append(patch("src.graph.pipeline.classify_followup", return_value=classifier))
        for item in patches:
            item.start()
        try:
            final = run_turn(graph, thread, question)
            snapshot = graph.get_state({"configurable": {"thread_id": thread}}).values
        finally:
            for item in reversed(patches):
                item.stop()
        return final, snapshot

    def test_standalone_comparison_runs_full_two_scope_path(self):
        final, snapshot = self._run(
            "Compare the current NovaCred and LumenTrail credit-freeze policies."
        )
        self.assertEqual("compare_policy", final["behavior"])
        self.assertEqual(2, len(final["comparison_scopes"]))
        self.assertEqual(10, len(final["context_chunks"]))
        self.assertEqual("policy_comparison", snapshot["history"][-1]["turn_type"])
        self.assertEqual(2, len(snapshot["history"][-1]["scopes"]))

    def test_missing_antecedent_pauses_before_any_search(self):
        graph = build_graph(correction_mode="off")
        with (
            patch("src.guardrails.checks.screen"),
            patch("src.graph.pipeline.qs.search_hybrid") as hybrid,
            patch("src.graph.runtime.tracing_is_enabled", return_value=False),
        ):
            final = run_turn(graph, "comparison-missing-antecedent", "Compare that with NovaCred.")
        self.assertTrue(final.get("__interrupt__"))
        self.assertEqual(
            "missing_comparison_target",
            final["__interrupt__"][0].value["reason"],
        )
        hybrid.assert_not_called()

    def test_followup_comparison_preserves_historical_antecedent(self):
        history = [{
            "question": "What does LumenTrail V0 require for identity evidence?",
            "answer": "Prior answer.",
            "chunk_ids": [],
            "scope": _scope("lumentrail", "V0"),
            "turn_type": "policy_answer",
        }]
        classifier = {
            "behavior": "compare_policy",
            "standalone": (
                "Compare LumenTrail Financial V0 identity-evidence policy with "
                "NovaCred Financial identity-evidence policy."
            ),
            "instruction": "",
        }
        final, _ = self._run(
            "Compare that with NovaCred.",
            history=history,
            prior_scope=_scope("lumentrail", "V0"),
            classifier=classifier,
        )
        versions = {scope["lender_id"]: scope["document_version"]
                    for scope in final["comparison_scopes"]}
        self.assertEqual({"lumentrail": "V0", "novacred": "V1"}, versions)


if __name__ == "__main__":
    unittest.main()
