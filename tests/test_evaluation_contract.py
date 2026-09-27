"""Offline evaluator contracts. Mocked verdicts test plumbing, NOT LLM accuracy.

Run with LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false.
Every test blocks network connections and the judge client.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.eval import judge
from src.eval.generation_metrics import aggregate_generation, score_generation
from scripts import run_retry_eval as retry
from scripts import run_conversation_eval as conversation


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("socket.socket.connect", side_effect=AssertionError("network forbidden")).start()
        patch.object(judge, "_client", side_effect=AssertionError("judge API forbidden")).start()


class ClaimChecklistTests(OfflineTest):
    def test_claim_rubric_preserves_irrelevance_as_an_atomic_assertion(self):
        self.assertIn("does or does not affect the result", judge.PROMPT)
        self.assertIn("do not paraphrase", judge.PROMPT.lower())

    def test_object_judges_disable_reasoning_and_request_json_mode(self):
        response = type("Response", (), {
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": '{"verdict":"addresses","evidence":""}'})()})()],
            "usage": type("Usage", (), {"total_tokens": 20})(),
        })()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None)))
        with tempfile.TemporaryDirectory() as cache, patch.object(judge, "_client", return_value=fake_client), \
             patch.object(judge, "tracked_call", return_value=response) as call:
            data = judge._ask("question and answer data", "mock-model", cache, "relev", 400,
                              system_prompt="return JSON")
        self.assertEqual("addresses", data["verdict"])
        self.assertEqual("none", call.call_args.kwargs["reasoning_effort"])
        self.assertEqual({"type": "json_object"}, call.call_args.kwargs["response_format"])
        self.assertEqual(["system", "user"],
                         [m["role"] for m in call.call_args.kwargs["messages"]])
        self.assertEqual("return JSON", call.call_args.kwargs["messages"][0]["content"])
        self.assertEqual("question and answer data", call.call_args.kwargs["messages"][1]["content"])

    def check(self, statuses):
        answer = "Retain NOAA 101. Retain NOAA 105."
        checks = [{"assertion": f"assertion {i}", "verdict": status,
                   "evidence": "Retain NOAA 101." if status != "absent" else ""}
                  for i, status in enumerate(statuses)]
        return judge._parse(json.dumps({"checks": checks, "verdict": "present"}), answer)

    def test_partial_compound_claim_does_not_pass(self):
        self.assertEqual(self.check(["present", "absent"])["verdict"], "absent")

    def test_contradicted_component_does_not_pass(self):
        self.assertEqual(self.check(["present", "contradicted"])["verdict"], "absent")

    def test_all_components_required(self):
        self.assertEqual(self.check(["present", "present"])["verdict"], "present")

    def test_unverifiable_quote_is_judge_error(self):
        data = {"checks": [{"assertion": "a", "verdict": "present", "evidence": "invented quote"}]}
        with self.assertRaises(ValueError):
            judge._parse(json.dumps(data), "Actual answer.")

    def test_empty_or_legacy_checklist_is_error(self):
        for data in ({"checks": []}, {"verdict": "present", "evidence": "yes"}, []):
            with self.subTest(data=data), self.assertRaises(ValueError):
                judge._parse(json.dumps(data), "yes")

    def test_prompt_change_changes_claim_cache_key(self):
        old = judge._key("model", "claim", "answer")
        with patch.object(judge, "PROMPT", judge.PROMPT + " changed"):
            self.assertNotEqual(old, judge._key("model", "claim", "answer"))

    def test_legacy_cache_cannot_supply_new_verdict(self):
        # Patch only the API client: absence of a new-schema cache must try the
        # blocked client, never read the old model/claim/answer cache entry.
        import hashlib
        with tempfile.TemporaryDirectory() as cache:
            legacy = hashlib.sha256(b"m|c|a").hexdigest()[:20]
            Path(cache, legacy + ".json").write_text('{"verdict":"present","evidence":"a"}')
            with self.assertRaises(AssertionError):
                judge.judge_claim("c", "a", model="m", cache_dir=cache)

    def test_claim_judge_repairs_an_unverifiable_evidence_span_once(self):
        def response(content):
            return type("Response", (), {
                "choices": [type("Choice", (), {
                    "message": type("Message", (), {"content": content})()
                })()],
                "usage": type("Usage", (), {"total_tokens": 20})(),
            })()

        invalid = json.dumps({"checks": [{
            "assertion": "The answer is actual.", "verdict": "present",
            "evidence": "invented passage",
        }]})
        repaired = json.dumps({"checks": [{
            "assertion": "The answer is actual.", "verdict": "present",
            "evidence": "Actual answer.",
        }]})
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))
        )
        with tempfile.TemporaryDirectory() as cache, \
             patch.object(judge, "_client", return_value=fake_client), \
             patch.object(judge, "tracked_call", side_effect=[response(invalid), response(repaired)]) as call:
            verdict, evidence = judge.judge_claim(
                "The answer is actual.", "Actual answer.", model="mock", cache_dir=cache
            )
        self.assertTrue(verdict)
        self.assertEqual("Actual answer.", evidence)
        self.assertEqual(2, call.call_count)
        self.assertEqual("judge_claim_repair", call.call_args.args[1])


class GroundingTests(OfflineTest):
    def test_support_rubric_allows_scenario_plus_policy_application(self):
        prompt = " ".join(judge.SUPPORT_BATCH_PROMPT.split())
        self.assertIn("direct result of combining", prompt)
        self.assertIn("Do not require", prompt)
        self.assertIn("the threshold is not met", prompt)

    def test_question_is_passed_separately_from_policy(self):
        question = "ID expires 20 days after funding; does 128 apply?"
        context = "At least 30 validity days must remain."
        with patch.object(judge, "_ask", return_value=[{"n": 1, "verdict": "supported"}]) as ask:
            self.assertEqual([True], judge.check_supported_batch(["20 is below 30"], context, question=question))
        prompt = ask.call_args.args[0]
        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
        self.assertIn("Do NOT treat the user's proposed code meaning",
                      ask.call_args.kwargs["system_prompt"])

    def test_changing_question_changes_support_prompt(self):
        with patch.object(judge, "_ask", return_value=[{"n": 1, "verdict": "supported"}]) as ask:
            judge.check_supported_batch(["claim"], "policy", question="20 days")
            first = ask.call_args.args[0]
            judge.check_supported_batch(["claim"], "policy", question="40 days")
            self.assertNotEqual(first, ask.call_args.args[0])

    def test_malformed_or_missing_support_is_not_scored_false(self):
        bad = [[], {}, [{"n": 1, "verdict": "maybe"}],
               [{"n": 2, "verdict": "supported"}],
               [{"n": True, "verdict": "supported"}]]
        for data in bad:
            with self.subTest(data=data), patch.object(judge, "_ask", return_value=data), self.assertRaises(ValueError):
                judge.check_supported_batch(["claim"], "context")

    def test_duplicate_indices_are_errors(self):
        data = [{"n": 1, "verdict": "supported"}] * 2
        with patch.object(judge, "_ask", return_value=data), self.assertRaises(ValueError):
            judge.check_supported_batch(["one", "two"], "context")

    def test_empty_claims_do_not_call_judge(self):
        with patch.object(judge, "_ask", side_effect=AssertionError("must not call")):
            self.assertEqual([], judge.check_supported_batch([], "context"))

    def test_bad_decomposition_is_not_empty_perfect_answer(self):
        with patch.object(judge, "_ask", return_value={"claims": []}), self.assertRaises(ValueError):
            judge.decompose_claims("answer")

    def test_retry_grading_passes_original_question(self):
        case = {"question": "20 days remaining", "required_claims": [], "forbidden_claims": []}
        with patch.object(retry, "decompose_claims", return_value=["20 < 30"]), patch.object(retry, "check_supported_batch", return_value=[True]) as check:
            retry.grade_answer(case, "answer", "30-day rule", "mock")
        self.assertEqual(case["question"], check.call_args.kwargs["question"])


class ResponseAndDenominatorTests(OfflineTest):
    def test_response_rubric_requires_one_verbatim_span(self):
        self.assertIn("shortest", judge.RESPONSE_PROMPT)
        self.assertIn("Do not paraphrase", judge.RESPONSE_PROMPT)
        self.assertIn("including Markdown characters", judge.RESPONSE_PROMPT)

    def test_result_schemas_allow_unmeasured_usage_and_response_annotations(self):
        from eval.retry_eval_v1.validate import check_schema as check_retry
        from eval.conversation_eval_v1.validate import check_schema as check_conversation
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "eval/retry_eval_v1/result_record.schema.json").read_text())
        check_retry({"elapsed_ms": 0, "provider_calls": None, "tokens": None, "cost_usd": None,
                     "cache_hits": None, "transport_retries": None},
                    schema["definitions"]["attempt"]["properties"]["usage"], schema)
        check_retry({"kind": "answer"}, schema["properties"]["response_assessment"], schema)
        other = json.loads((root / "eval/conversation_eval_v1/result_record.schema.json").read_text())
        check_conversation({"kind": "answer"}, other["properties"]["response_assessment"], other)

    def test_unmeasured_relevance_is_not_a_perfect_score(self):
        unmeasured = score_generation({"id": "x", "question": "q"}, "a", [], lambda c, a: (True, a))
        measured = score_generation({"id": "y", "question": "q"}, "a", [], lambda c, a: (True, a),
                                    relevance=lambda q, a: False)
        self.assertIsNone(unmeasured.addresses_question)
        self.assertIsNone(aggregate_generation([unmeasured])["answer_relevance"])
        aggregate = aggregate_generation([unmeasured, measured])
        self.assertEqual(0, aggregate["answer_relevance"])
        self.assertEqual(1, aggregate["metric_denominators"]["addresses_question"])

    def test_abstention_is_not_clarification_or_confidence(self):
        answer = "I cannot determine what QXR means from this material."
        verdict = {"kind": "abstention", "evidence": answer, "rationale": "no answer given"}
        with patch.object(judge, "_ask", return_value=verdict):
            observed = judge.classify_response("Can QXR clear it?", answer)
        self.assertEqual(observed["kind"], "abstention")
        self.assertFalse(judge.response_type_pass("clarify", observed["kind"]))
        self.assertFalse(judge.response_type_pass("answer", observed["kind"]))

    def test_qualified_answer_requires_visible_limitation(self):
        self.assertFalse(judge.response_type_pass("qualified_answer", "answer"))
        self.assertTrue(judge.response_type_pass("qualified_answer", "qualified_answer"))
        self.assertTrue(judge.response_type_pass("clarify", "clarification"))

    def test_invalid_response_quote_is_error(self):
        with patch.object(judge, "_ask", return_value={"kind": "answer", "evidence": "invented"}), self.assertRaises(ValueError):
            judge.classify_response("q", "a")

    def test_response_quote_may_omit_only_markdown_presentation_markers(self):
        answer = "Action: **RFAI** (Request for Additional Information)"
        verdict = {
            "kind": "answer",
            "evidence": "RFAI (Request for Additional Information)",
            "rationale": "substantive answer",
        }
        with patch.object(judge, "_ask", return_value=verdict):
            observed = judge.classify_response("What is the action?", answer)
        self.assertEqual("answer", observed["kind"])

    def test_rendered_quote_cannot_join_separate_answer_passages(self):
        answer = "**RFAI** is used here. Other text. Code **105** applies."
        verdict = {
            "kind": "answer", "evidence": "RFAI Code 105", "rationale": "joined",
        }
        with patch.object(judge, "_ask", return_value=verdict), self.assertRaises(ValueError):
            judge.classify_response("q", answer)

    def test_no_citations_or_grounding_measurement_is_na(self):
        score = score_generation({"id": "test", "question": "q"}, "answer", [], lambda c, a: (True, a))
        self.assertIsNone(score.citation_validity)
        self.assertIsNone(score.groundedness)
        self.assertIsNone(score.evidence_complete)
        self.assertIsNone(score.over_refusal)
        aggregate = aggregate_generation([score])
        self.assertIsNone(aggregate["groundedness"])
        self.assertEqual(0, aggregate["metric_denominators"]["groundedness"])
        self.assertEqual(0, aggregate["cases_missing_evidence"])

    def test_short_support_callback_is_error(self):
        with self.assertRaises(ValueError):
            score_generation({"id": "test", "question": "q"}, "a", [], lambda c, a: (True, a),
                             context="p", decompose=lambda a: ["a", "b"], supported=lambda cs, ctx: [True])

    def test_clarification_stop_excluded_from_retrieval_denominator(self):
        case = {"expected_response": "answer"}
        oracle = {"covered_groups": [], "missing_groups": ["rule"]}
        metrics = retry.build_metrics(case, "clarification", [], {"made": 0}, oracle, None)
        self.assertIsNone(metrics["context_group_coverage"])
        self.assertFalse(metrics["expected_response_pass"])
        self.assertTrue(metrics["pipeline_success"])  # execution only, NOT quality

    def test_completed_does_not_imply_expected_answer(self):
        metrics = retry.build_metrics({"expected_response": "answer"}, "completed", [], {"made": 0},
                                      {"covered_groups": ["rule"], "missing_groups": []}, None,
                                      {"kind": "abstention"})
        self.assertFalse(metrics["expected_response_pass"])
        self.assertEqual(1.0, metrics["context_group_coverage"])

    def test_missing_response_judgment_is_not_pass(self):
        metrics = retry.build_metrics({"expected_response": "answer"}, "completed", [], {"made": 0}, {}, None)
        self.assertIsNone(metrics["expected_response_pass"])

    def test_clarification_run_records_no_retrieval_verdict(self):
        case = {"id": "stop", "question": "q", "expected_response": "answer",
                "evidence_groups": [{"name": "rule", "any_of_source_refs": ["ref"]}]}
        with patch.object(retry, "run_turn", return_value={"__interrupt__": ["Which lender?"]}):
            record = retry.run_case(None, case, "r", {"ref": "id"}, "mock", {})
        self.assertEqual("not_applicable", record["attempts"][0]["oracle"]["evidence_status"])
        self.assertFalse(record["metrics"]["expected_response_pass"])

    def test_conversation_interrupt_is_scored_as_clarify(self):
        case = {"id": "c", "difficulty": "easy", "scenario_family": "test",
                "_run_id": "r", "_thread_ids": {"A": "a"}}
        step = {"step_id": "s", "thread": "A", "user_message": "Which rule?",
                "expected": {"behavior": "clarify", "scope": None,
                             "evidence_policy": "clarification_required", "required_claims": [], "forbidden_claims": []}}
        record = conversation.score_step(case, step, {"__interrupt__": ["Which lender?"]}, {}, "mock")
        self.assertTrue(record["metrics"]["behavior_pass"])
        self.assertIsNone(record["response_assessment"])

    def test_conversation_uses_original_input_and_history(self):
        case = {"id": "c", "difficulty": "easy", "scenario_family": "test",
                "_run_id": "r", "_thread_ids": {"A": "a"}}
        step = {"step_id": "s", "thread": "A", "user_message": "Explain that without removing the exception",
                "expected": {"behavior": "transform_previous_answer", "scope": None,
                             "evidence_policy": "stored_evidence_sufficient", "required_claims": ["claim"], "forbidden_claims": []}}
        final = {"resolved_question": "bad rewrite", "answer": "answer", "history": [{"question": "original scenario", "answer": "prior answer"}],
                 "behavior": "transform_previous_answer", "evidence_reused": True, "context_chunks": [{"chunk_id": "id", "text": "policy"}]}
        with patch.object(conversation, "classify_response", return_value={"kind": "answer"}), patch.object(conversation, "judge_claim", return_value=(True, "answer")), patch.object(conversation, "decompose_claims", return_value=["claim"]), patch.object(conversation, "check_supported_batch", return_value=[True]) as support, patch.object(conversation, "check_relevance", return_value=True):
            record = conversation.score_step(case, step, final, {}, "mock")
        actual_input = support.call_args.kwargs["question"]
        self.assertIn(step["user_message"], actual_input)
        self.assertIn("original scenario", actual_input)
        self.assertNotIn("bad rewrite", actual_input)
        self.assertTrue(record["metrics"]["behavior_pass"])
        self.assertIsNone(record["metrics"]["citation_validity"])


if __name__ == "__main__":
    unittest.main()
