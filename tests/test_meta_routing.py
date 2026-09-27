"""Meta questions use the manifest or checkpoint, never manual retrieval."""

from __future__ import annotations

import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.graph.pipeline import build_graph
from src.graph.answer_review import request_input
from src.graph.runtime import run_turn, seed_thread
from src.resolve.meta import detect_meta_intent, latest_lender_answer, lender_inventory


OLD_LUMENTRAIL_TURN = {
    "question": "For LumenTrail V1, what happens after a bureau freeze?",
    "answer": "Evidence Required, code 110.",
    "chunk_ids": ["LumenTrail_UW_V1_pdf.pdf::0006"],
}
LUMENTRAIL_SCOPE = {
    "status": "resolved", "lender_id": "lumentrail",
    "lender_name": "LumenTrail Financial", "document_version": "V1",
}


class MetaIntentTests(unittest.TestCase):
    def test_explicit_catalog_and_conversation_phrasings(self):
        for question in ("which are all the lenders in corpus", "What lenders do you cover?",
                         "List lenders in the manuals"):
            with self.subTest(question=question):
                self.assertEqual("list_corpus_lenders", detect_meta_intent(question))
        for question in ("what is the latest lender answer returned",
                         "Which lender was the most recent answer about?"):
            with self.subTest(question=question):
                self.assertEqual("last_lender_answer", detect_meta_intent(question))

    def test_policy_questions_are_not_diverted(self):
        for question in ("What do both lenders say about credit freeze?",
                         "Which lenders in the corpus use code 110?",
                         "What was your latest answer about the prescreen rule?",
                         "What is your latest answer for LumenTrail?"):
            with self.subTest(question=question):
                self.assertIsNone(detect_meta_intent(question))

    def test_inventory_comes_from_supplied_manifest_specs(self):
        specs = [SimpleNamespace(lender_name="Second Bank"),
                 SimpleNamespace(lender_name="First Bank"),
                 SimpleNamespace(lender_name="First Bank")]
        self.assertEqual("The underwriting corpus covers First Bank and Second Bank.",
                         lender_inventory(specs))

    def test_structured_history_is_preferred_and_meta_turns_are_skipped(self):
        history = [{"question": "policy", "answer": "result", "chunk_ids": ["a"],
                    "turn_type": "policy_answer", "scope": LUMENTRAIL_SCOPE},
                   {"question": "list", "answer": "both lenders", "chunk_ids": [],
                    "turn_type": "list_corpus_lenders", "scope": {"status": "not_applicable"}}]
        answer, basis = latest_lender_answer(history)
        self.assertIn("LumenTrail Financial V1", answer)
        self.assertIn("preceding reply", answer)
        self.assertEqual({"kind": "conversation_history", "turn_index": 0}, basis)

    def test_legacy_history_uses_only_unambiguous_manifest_chunk_ids(self):
        answer, basis = latest_lender_answer([OLD_LUMENTRAIL_TURN])
        self.assertIn("LumenTrail Financial V1", answer)
        self.assertEqual(0, basis["turn_index"])
        mixed = {**OLD_LUMENTRAIL_TURN,
                 "chunk_ids": ["LumenTrail_UW_V1_pdf.pdf::0006", "NovaCred_UW_V1_pdf.pdf::0007"]}
        answer, basis = latest_lender_answer([mixed])
        self.assertIn("not given", answer)
        self.assertIsNone(basis["turn_index"])

    def test_legacy_misrouted_meta_answer_is_not_treated_as_policy(self):
        # This is the exact shape of the pre-change turn: it searched a manual
        # and acquired chunk IDs even though the question was about the chat.
        misrouted = {"question": "Which lender's answer was returned most recently in the conversation? lumentrail",
                     "answer": "The last answer pertains to LumenTrail.",
                     "chunk_ids": ["LumenTrail_UW_V1_pdf.pdf::0000"]}
        answer, basis = latest_lender_answer([OLD_LUMENTRAIL_TURN, misrouted])
        self.assertIn("LumenTrail Financial V1", answer)
        self.assertEqual(0, basis["turn_index"])

    def test_review_prompt_does_not_include_history_audit_metadata(self):
        payload = json.loads(request_input({
            "question": "What does the rule say?", "clarifications": [],
            "history": [{**OLD_LUMENTRAIL_TURN, "turn_type": "policy_answer",
                         "scope": LUMENTRAIL_SCOPE, "answer_basis": {"kind": "checkpoint"}}],
        }))
        self.assertEqual([OLD_LUMENTRAIL_TURN], payload["prior_conversation"])


class MetaGraphTests(unittest.TestCase):
    def setUp(self):
        guard = patch("src.guardrails.checks._semantic_call",
                      return_value='{"decision":"allow","reason":"ordinary_content"}')
        guard.start()
        self.addCleanup(guard.stop)
        self.graph = build_graph(1, correction_mode="post_generation")

    def test_catalog_then_last_lender_after_legacy_checkpoint(self):
        thread_id = "legacy-meta"
        seed_thread(self.graph, thread_id, [OLD_LUMENTRAIL_TURN], LUMENTRAIL_SCOPE)
        with (patch("src.graph.pipeline.classify_followup", side_effect=AssertionError("no LLM classifier")),
              patch("src.graph.pipeline.qs.connect", side_effect=AssertionError("no Qdrant")),
              patch("src.graph.pipeline.decompose_query", side_effect=AssertionError("no decomposition")),
              patch("src.graph.pipeline.review_draft", side_effect=AssertionError("no reviewer"))):
            catalog = run_turn(self.graph, thread_id, "which are all the lenders in corpus")
            latest = run_turn(self.graph, thread_id, "what is the latest lender answer returned")

        self.assertEqual("list_corpus_lenders", catalog["behavior"])
        self.assertEqual("The underwriting corpus covers LumenTrail Financial and NovaCred Financial.",
                         catalog["answer"])
        self.assertEqual(["begin_turn", "contextualize", "catalog_lenders"],
                         [event["node"] for event in catalog["node_events"]])
        self.assertEqual("corpus_manifest", catalog["answer_basis"]["kind"])
        self.assertEqual("last_lender_answer", latest["behavior"])
        self.assertIn("LumenTrail Financial V1", latest["answer"])
        self.assertNotIn("【", latest["answer"])
        self.assertEqual("conversation_history", latest["answer_basis"]["kind"])
        self.assertEqual(["begin_turn", "contextualize", "conversation_history"],
                         [event["node"] for event in latest["node_events"]])
        saved = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values["history"]
        self.assertEqual(3, len(saved))
        self.assertEqual("list_corpus_lenders", saved[1]["turn_type"])
        self.assertEqual("not_applicable", saved[1]["scope"]["status"])

    def test_empty_history_does_not_invent_a_lender(self):
        with patch("src.graph.pipeline.qs.connect", side_effect=AssertionError("no Qdrant")):
            final = run_turn(self.graph, "empty-meta", "what is the latest lender answer returned")
        self.assertIn("not given", final["answer"])
        self.assertIsNone(final["answer_basis"]["turn_index"])


if __name__ == "__main__":
    unittest.main()
