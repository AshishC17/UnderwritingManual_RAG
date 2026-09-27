"""Every chat-model call keeps stable instructions above untrusted data."""
from __future__ import annotations

import os
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.decompose import decomposer
from src.generate import generator
from src.ingest import flowchart
from src.resolve import followup


def completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def inert_groq_client():
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None)))


class PromptRoleTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def assert_boundary(self, call, expected_system: str, user_fragment: str):
        messages = call.call_args.kwargs["messages"]
        self.assertEqual(["system", "user"], [message["role"] for message in messages])
        self.assertEqual(expected_system, messages[0]["content"])
        self.assertIn(user_fragment, messages[1]["content"])
        self.assertNotIn(user_fragment, messages[0]["content"])

    def test_query_decomposition_separates_rules_from_question(self):
        with tempfile.TemporaryDirectory() as cache, \
             patch.object(decomposer, "_client", return_value=inert_groq_client()), \
             patch.object(decomposer, "tracked_call", return_value=completion('["SECRET QUESTION"]')) as call:
            decomposer.decompose_query("SECRET QUESTION", cache_dir=cache)
        self.assert_boundary(call, decomposer.PROMPT, "SECRET QUESTION")

    def test_query_reformulation_separates_rules_from_queries(self):
        with tempfile.TemporaryDirectory() as cache, \
             patch.object(decomposer, "_client", return_value=inert_groq_client()), \
             patch.object(decomposer, "tracked_call", return_value=completion('{"query":"rewritten"}')) as call:
            decomposer.reformulate_query("SECRET SEED", "SECRET ORIGINAL", cache_dir=cache)
        self.assert_boundary(call, decomposer.REFORMULATE_PROMPT, "SECRET ORIGINAL")
        self.assertIn("SECRET SEED", call.call_args.kwargs["messages"][1]["content"])

    def test_followup_classification_separates_rules_from_history(self):
        response = '{"behavior":"transform_previous_answer","standalone":"NovaCred V1 topic","instruction":"SECRET FOLLOWUP"}'
        scope = {"status": "resolved", "lender_name": "NovaCred Financial", "document_version": "V1"}
        history = [{"question": "SECRET HISTORY", "answer": "previous"}]
        with tempfile.TemporaryDirectory() as cache, \
             patch.object(followup, "corpus_lenders", return_value=["NovaCred Financial"]), \
             patch.object(followup, "_client", return_value=inert_groq_client()), \
             patch.object(followup, "tracked_call", return_value=completion(response)) as call:
            followup.classify_followup("SECRET FOLLOWUP", history, scope, cache_dir=cache)
        self.assert_boundary(call, followup.PROMPT, "SECRET HISTORY")
        self.assertIn("SECRET FOLLOWUP", call.call_args.kwargs["messages"][1]["content"])

    def test_generation_already_separates_rules_from_context(self):
        chunk = {"chunk_id": "manual::0001", "section": "Rules", "text": "SECRET EVIDENCE"}
        with tempfile.TemporaryDirectory() as cache, \
             patch.object(generator, "_client", return_value=inert_groq_client()), \
             patch.object(generator, "tracked_call", return_value=completion("answer")) as call:
            generator.generate("SECRET QUESTION", [chunk], cache_dir=cache)
        self.assert_boundary(call, generator.SYSTEM, "SECRET EVIDENCE")
        self.assertIn("SECRET QUESTION", call.call_args.kwargs["messages"][1]["content"])

    def test_flowchart_caption_separates_rules_from_image_payload(self):
        fake_messages = MagicMock()
        fake_messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="caption")]
        )
        fake_client = SimpleNamespace(messages=fake_messages)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"synthetic-image")
            ref = flowchart.ImageRef(image, 1, "sha")
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}), \
                 patch("anthropic.Anthropic", return_value=fake_client), \
                 patch.object(flowchart, "policy", return_value=flowchart.policy().model_copy(update={
                     "reviewed_image_sha256": [hashlib.sha256(b"synthetic-image").hexdigest()]})):
                self.assertEqual("caption", flowchart._call_vision(ref))
        kwargs = fake_messages.create.call_args.kwargs
        self.assertEqual(flowchart.CAPTION_PROMPT, kwargs["system"])
        self.assertEqual("user", kwargs["messages"][0]["role"])
        self.assertNotIn(flowchart.CAPTION_PROMPT, str(kwargs["messages"]))


if __name__ == "__main__":
    unittest.main()
