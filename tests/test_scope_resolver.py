from __future__ import annotations

import unittest
from unittest.mock import patch

from langgraph.types import Command

from src.graph.pipeline import build_graph, retrieve_node
from src.ingest.manifest import load_manifest
from src.resolve.scope import resolve_query_scope


class ScopeResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = load_manifest("config/corpus_manifest.json", require_files=False)

    def test_lender_without_date_or_version_defaults_to_current(self):
        result = resolve_query_scope(
            "What is NovaCred's minimum income requirement?", self.specs
        )

        self.assertEqual("resolved", result["status"])
        self.assertEqual("default_current", result["reason"])
        self.assertEqual("novacred", result["lender_id"])
        self.assertEqual("V1", result["document_version"])
        self.assertTrue(result["filter_args"]["current_only"])

    def test_as_of_date_selects_historical_version(self):
        result = resolve_query_scope(
            "For NovaCred on June 15, 2026, what was the income requirement?",
            self.specs,
        )

        self.assertEqual("as_of_date", result["reason"])
        self.assertEqual("V0", result["document_version"])
        self.assertEqual("2026-06-15", result["filter_args"]["as_of"])

    def test_explicit_version_selects_that_version(self):
        result = resolve_query_scope(
            "What did LumenTrail V0 say about Code 128?", self.specs
        )

        self.assertEqual("explicit_version", result["reason"])
        self.assertEqual("V0", result["document_version"])
        self.assertEqual("lumentrail", result["lender_id"])

    def test_missing_lender_routes_to_clarification(self):
        result = resolve_query_scope("What does Code 128 mean?", self.specs)

        self.assertEqual("clarification", result["status"])
        self.assertEqual("missing_lender", result["reason"])

    def test_conflicting_version_and_date_routes_to_clarification(self):
        result = resolve_query_scope(
            "What did NovaCred V0 require on 2026-07-15?", self.specs
        )

        self.assertEqual("clarification", result["status"])
        self.assertEqual("version_date_conflict", result["reason"])

    def test_partial_date_does_not_silently_default_to_current(self):
        result = resolve_query_scope(
            "What was NovaCred's requirement in June 2026?", self.specs
        )

        self.assertEqual("clarification", result["status"])
        self.assertEqual("partial_date", result["reason"])

    def test_multiple_lenders_do_not_get_one_blended_filter(self):
        result = resolve_query_scope(
            "Compare NovaCred with LumenTrail.", self.specs
        )

        self.assertEqual("clarification", result["status"])
        self.assertEqual("multiple_lenders", result["reason"])


class GraphScopeConnectionTests(unittest.TestCase):
    def test_graph_stops_before_retrieval_when_lender_is_missing(self):
        """An unscoped question pauses for input rather than answering.

        The graph interrupts instead of routing to END, so there is no answer
        to assert on — the absence of retrieval is the assertion.
        """
        app = build_graph()
        config = {"configurable": {"thread_id": "test-missing-lender"}}
        final = app.invoke({"question": "What does Code 128 mean?"}, config=config)

        self.assertEqual("missing_lender", final["scope"]["reason"])
        self.assertEqual([], final.get("context_chunks", []))

        interrupts = final["__interrupt__"]
        self.assertEqual(1, len(interrupts))
        self.assertIn("Which lender", interrupts[0].value["question"])
        self.assertEqual(("clarify",), app.get_state(config).next)

    def test_clarification_folds_into_one_complete_question(self):
        """A clarification reply is merged, not stored as its own question.

        Everything downstream of the resolver is stubbed: this test is about
        the fold and the re-resolution, and it must not reach an API.
        """
        app = build_graph()
        config = {"configurable": {"thread_id": "test-resume"}}
        with (
            patch("src.graph.pipeline.decompose_query", side_effect=lambda q: [q]),
            patch("src.graph.pipeline.qs.connect", return_value=object()),
            patch("src.graph.pipeline.embed_query_scoped", return_value=[0.0]),
            patch("src.graph.pipeline.embed_query_sparse", return_value=object()),
            patch("src.graph.pipeline.qs.search_hybrid", return_value=[]),
            patch("src.graph.pipeline.rerank", return_value=[]),
            patch("src.graph.pipeline.generate", return_value="stub answer"),
        ):
            app.invoke({"question": "What does Code 128 mean?"}, config=config)
            final = app.invoke(Command(resume="LumenTrail V0"), config=config)

        self.assertEqual("resolved", final["scope"]["status"])
        self.assertEqual("lumentrail", final["scope"]["lender_id"])
        self.assertEqual("V0", final["scope"]["document_version"])
        self.assertEqual(1, final["clarification_rounds"])
        self.assertEqual(1, len(final["clarifications"]))
        self.assertTrue(
            final["resolved_question"].endswith("LumenTrail V0"),
            final["resolved_question"],
        )

    def test_retrieve_passes_resolved_filter_to_every_subquery(self):
        state = {
            "sub_queries": ["first", "second"],
            "scope": {
                "status": "resolved",
                "filter_args": {
                    "lender_id": "novacred",
                    "document_id": "novacred_flex_uw",
                    "document_version": "V1",
                    "current_only": True,
                },
            },
        }
        sentinel_filter = object()

        with (
            patch("src.graph.pipeline.qs.connect", return_value=object()),
            patch("src.graph.pipeline.qs.scope_filter", return_value=sentinel_filter),
            patch("src.graph.pipeline.embed_query_scoped", return_value=[0.0]),
            patch("src.graph.pipeline.embed_query_sparse", return_value=object()),
            patch("src.graph.pipeline.qs.search_hybrid", return_value=[]) as search,
        ):
            result = retrieve_node(state)

        self.assertEqual({}, result["retrieved_candidates"])
        self.assertEqual(2, search.call_count)
        self.assertTrue(
            all(call.kwargs["query_filter"] is sentinel_filter for call in search.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
