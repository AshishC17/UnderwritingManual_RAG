"""Real graph transitions, fake external models/tools. NOT model quality scores."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.graph import pipeline as p
from src.graph import answer_review as ar
from src.graph.runtime import run_turn, seed_thread
from src.resolve.scope import resolve_query_scope
from src.util import telemetry as t
from src.util import groq_ratelimit as gr
from src.generate import generator as gen
from scripts.run_retry_eval import build_attempt

QUESTION = "For NovaCred V1, explain the rule and its exception."
SCOPE = resolve_query_scope(QUESTION)


def chunk(cid="rule", **overrides):
    return {"chunk_id": cid, "text": "The rule applies. An exception needs approval.",
            "token_count": 12, "lender_id": SCOPE["lender_id"],
            "document_id": SCOPE["document_id"], "document_version": SCOPE["document_version"], **overrides}


def hit(c):
    return SimpleNamespace(payload=c, score=0.8)


def verdict(evidence="present", answer="covered", *, clarify=False):
    return ar.validate_review({
        "requirements": [{"need": "Explain the exception", "evidence": evidence,
                          "answer": answer, "source_ids": ["rule"] if evidence == "present" else []}],
        "unsupported_claims": [], "needs_clarification": clarify,
        "clarification_question": "Which stage?" if clarify else "", "reason": "fixture diagnosis",
        "supported_excerpts": [{"chunk_id": "rule", "quote": "The rule applies."}],
    }, [chunk()])


class GraphCorrectionTests(unittest.TestCase):
    def setUp(self):
        def mocked(target, **kwargs):
            manager = patch(target, **kwargs)
            self.addCleanup(manager.stop)
            return manager.start()
        mocked("socket.socket.connect", side_effect=AssertionError("network forbidden"))
        mocked("src.guardrails.checks._semantic_call", return_value='{"decision":"allow","reason":"ordinary_content"}')
        mocked("src.graph.pipeline.classify_followup", side_effect=lambda q, h, **kw:
               {"behavior": "new_question", "standalone": q, "instruction": q})
        mocked("src.graph.pipeline.decompose_query", side_effect=lambda q: [q])
        mocked("src.graph.pipeline.qs.connect", return_value=object())
        mocked("src.graph.pipeline.embed_query_scoped", return_value=[0.0])
        mocked("src.graph.pipeline.embed_query_sparse", return_value=object())
        self.search = mocked("src.graph.pipeline.qs.search_hybrid", return_value=[hit(chunk())])
        self.rerank = mocked("src.graph.pipeline.rerank", side_effect=lambda q, chunks, **kw:
                             [(c, 0.9 - i * 0.01) for i, c in enumerate(chunks)])
        self.generate = mocked("src.graph.pipeline.generate", return_value="DRAFT")
        self.revise = mocked("src.graph.pipeline.revise_answer", return_value="REPAIRED")
        self.review = mocked("src.graph.pipeline.review_draft", return_value=verdict())
        self.graph = p.build_graph(1, correction_mode="post_generation")

    def run_case(self, question=QUESTION, thread="test"):
        return run_turn(self.graph, thread, question)

    def test_accept_has_no_recovery_and_complete_route_trace(self):
        final = self.run_case()
        self.assertEqual("DRAFT", final["answer"])
        self.assertEqual("accepted", final["final_status"])
        self.assertEqual(0, final["retry_attempt"])
        self.revise.assert_not_called()
        self.assertEqual(["begin_turn", "contextualize", "resolve_scope", "decompose", "retrieve", "rerank",
                          "check_evidence", "generate", "review_answer", "finalize"],
                         [n["node"] for n in final["node_events"]])
        self.assertEqual(1, self.search.call_count)

    def test_revision_uses_same_evidence_then_is_reviewed_again(self):
        self.review.side_effect = [verdict(answer="missing"), verdict()]
        final = self.run_case()
        self.assertEqual("REPAIRED", final["answer"])
        self.assertEqual("accepted", final["final_status"])
        self.assertEqual(1, self.search.call_count)
        self.assertEqual(1, self.revise.call_count)
        self.assertEqual(2, self.review.call_count)
        self.assertEqual([chunk()], self.revise.call_args.args[2])
        self.assertEqual(final["review_log"][0]["context"], final["review_log"][1]["context"])

    def test_failed_revision_stops_and_withholds_failed_draft(self):
        self.review.return_value = verdict(answer="wrong")
        final = self.run_case()
        self.assertEqual(1, final["retry_attempt"])
        self.assertEqual(1, self.revise.call_count)
        self.assertEqual("limited", final["final_status"])
        self.assertTrue(final["recovery_exhausted"])
        self.assertNotIn("REPAIRED", final["answer"])
        self.assertIn('"The rule applies." [rule]', final["answer"])

    def test_missing_evidence_retrieves_scoped_then_merges_and_reviews(self):
        self.review.side_effect = [verdict(evidence="missing", answer="missing"), verdict()]
        self.search.side_effect = [[hit(chunk())], [hit(chunk("exception"))]]
        final = self.run_case()
        self.assertEqual("accepted", final["final_status"])
        self.assertEqual(["targeted_retrieve"], final["tried_actions"])
        self.assertEqual({"rule", "exception"}, set(final["retrieved_candidates"]))
        self.assertEqual([], final["merge_lost_chunk_ids"])
        first, second = self.search.call_args_list
        self.assertEqual(first.kwargs["query_filter"], second.kwargs["query_filter"])
        self.assertEqual(30, second.kwargs["limit"])
        self.assertTrue(final["recovery_seed"].startswith(QUESTION))
        self.assertIn("Explain the exception", final["recovery_seed"])
        self.assertEqual(2, len(self.revise.call_args.args[2]))

    def test_second_missing_verdict_does_not_trigger_second_search_retry(self):
        self.review.return_value = verdict(evidence="missing", answer="missing")
        final = self.run_case()
        self.assertEqual("limited", final["final_status"])
        self.assertEqual(2, self.search.call_count)  # initial + one recovery
        self.assertEqual(1, self.revise.call_count)

    def test_empty_initial_context_retries_before_any_draft(self):
        self.search.side_effect = [[], [hit(chunk())]]
        final = self.run_case()
        self.assertEqual("accepted", final["final_status"])
        self.generate.assert_not_called()
        self.assertEqual(1, self.revise.call_count)
        self.assertEqual("evidence_guard", final["review_log"][0]["stage"])

    def test_persistently_empty_context_never_calls_generator_or_reviewer(self):
        self.search.return_value = []
        final = self.run_case()
        self.assertEqual("limited", final["final_status"])
        self.assertEqual(2, self.search.call_count)
        self.generate.assert_not_called()
        self.revise.assert_not_called()
        self.review.assert_not_called()

    def test_wrong_lender_or_version_stops_before_draft(self):
        for mismatch in ({"lender_id": "lumentrail"}, {"document_version": "V0"}, {"document_id": "wrong"}):
            with self.subTest(mismatch=mismatch):
                self.search.return_value = [hit(chunk(**mismatch))]
                final = self.run_case(thread=str(mismatch))
                self.assertEqual("scope_error", final["final_status"])
                self.assertEqual(0, final["retry_attempt"])
                self.assertFalse(final["recovery_exhausted"])
        self.generate.assert_not_called()
        self.review.assert_not_called()

    def test_recovery_out_of_scope_results_are_rejected(self):
        self.review.return_value = verdict(evidence="missing", answer="missing")
        self.search.side_effect = [[hit(chunk())], [hit(chunk("bad", document_version="V0"))]]
        final = self.run_case()
        self.assertEqual("limited", final["final_status"])
        self.assertNotIn("bad", final["retrieved_candidates"])
        self.assertTrue(final["correction_error"].startswith("retrieval:"))
        self.revise.assert_not_called()

    def test_review_error_is_not_automatic_accept(self):
        self.review.side_effect = ValueError("bad review")
        final = self.run_case()
        self.assertEqual("review_error", final["final_status"])
        self.assertNotIn("DRAFT", final["answer"])
        self.assertEqual(0, final["retry_attempt"])

    def test_revision_provider_error_is_bounded(self):
        self.review.return_value = verdict(answer="wrong")
        self.revise.side_effect = TimeoutError("API timeout")
        final = self.run_case()
        self.assertEqual("limited", final["final_status"])
        self.assertEqual("revision:TimeoutError", final["correction_error"])
        self.assertEqual(1, self.revise.call_count)
        self.assertEqual(1, self.review.call_count)

    def test_generation_provider_error_is_not_judged_as_quality_failure(self):
        self.generate.side_effect = TimeoutError("API timeout")
        final = self.run_case()
        self.assertEqual("generation:TimeoutError", final["correction_error"])
        self.review.assert_not_called()
        self.assertNotEqual("accepted", final["final_status"])

    def test_clarification_interrupt_and_resume_preserve_original_request(self):
        self.review.side_effect = [verdict(clarify=True), verdict()]
        first = self.run_case()
        self.assertEqual("Which stage?", first["__interrupt__"][0].value["question"])
        second = self.run_case("Prescreen")
        self.assertEqual("accepted", second["final_status"])
        self.assertEqual(QUESTION, second["question"])
        self.assertEqual("Prescreen", second["clarifications"][0]["reply"])
        self.assertEqual(0, second["retry_attempt"])

    def test_repeated_semantic_clarification_has_separate_bound(self):
        self.review.return_value = verdict(clarify=True)
        self.assertIn("__interrupt__", self.run_case())
        self.assertIn("__interrupt__", self.run_case("Prescreen"))
        final = self.run_case("Prescreen, again")
        self.assertNotIn("__interrupt__", final)
        self.assertEqual(2, final["clarification_rounds"])
        self.assertEqual("limited", final["final_status"])

    def test_new_turn_resets_attempts_logs_and_context_but_keeps_history(self):
        self.review.side_effect = [verdict(answer="wrong"), verdict(), verdict()]
        first = self.run_case()
        self.assertEqual(1, first["retry_attempt"])
        self.search.return_value = [hit(chunk("new"))]
        second = self.run_case("For NovaCred V1, explain a different rule.")
        self.assertEqual(0, second["retry_attempt"])
        self.assertEqual(1, len(second["review_log"]))
        self.assertEqual({"new"}, set(second["retrieved_candidates"]))
        self.assertEqual(1, len(second["history"]))
        self.assertEqual(0, second["usage_summary"]["prior_completed_nodes"])

    def test_reuse_path_is_reviewed_without_search(self):
        seed_thread(self.graph, "reuse", [{"question": QUESTION, "answer": "old", "chunk_ids": ["rule"]}], SCOPE)
        with patch.object(p, "classify_followup", return_value={"behavior": "transform_previous_answer", "standalone": QUESTION, "instruction": "Simplify"}), \
             patch.object(p.qs, "get_by_chunk_ids", return_value=[chunk()]), \
             patch.object(p, "regenerate_from_evidence", return_value={"answer": "simple", "corrected_prior_claim": False}):
            final = self.run_case("Simplify", thread="reuse")
        self.assertTrue(final["evidence_reused"])
        self.assertEqual("accepted", final["final_status"])
        self.search.assert_not_called()
        self.review.assert_called_once()

    def test_recovery_preserves_all_pool_ids_with_bounded_shortlist(self):
        self.review.side_effect = [verdict(evidence="missing", answer="missing"), verdict()]
        self.search.side_effect = [[hit(chunk(str(i))) for i in range(15)],
                                   [hit(chunk(str(i))) for i in range(10, 40)]]
        final = self.run_case()
        self.assertEqual(40, len(final["retrieved_candidates"]))
        self.assertLessEqual(len(final["candidates"]), 24)
        self.assertLessEqual(len(final["rerank_assignments"]), 24)
        self.assertLessEqual(len(final["context_chunks"]), 10)
        self.assertEqual([], final["merge_lost_chunk_ids"])

    def test_baseline_does_not_run_review(self):
        self.graph = p.build_graph()
        final = self.run_case()
        self.assertEqual("DRAFT", final["answer"])
        self.review.assert_not_called()
        self.revise.assert_not_called()

    def test_one_attempt_cap_is_enforced_at_configuration(self):
        with self.assertRaises(ValueError):
            p.build_graph(2, correction_mode="post_generation")


class ReviewContractTests(unittest.TestCase):
    def test_reuse_cache_changes_when_prompt_or_text_changes(self):
        args = ("m", "simplify", "old answer", "q", ["same-id"])
        first = gen._reuse_key(*args, context="old rule")
        self.assertNotEqual(first, gen._reuse_key(*args, context="changed rule"))
        with patch.object(gen, "REUSE_SYSTEM", gen.REUSE_SYSTEM + " changed"):
            self.assertNotEqual(first, gen._reuse_key(*args, context="old rule"))

    def test_action_is_derived_not_trusted(self):
        data = verdict(answer="wrong")
        data["action"] = "accept"
        self.assertEqual("revise", ar.validate_review(data, [chunk()])["action"])

    def test_unknown_sources_and_empty_requirements_rejected(self):
        for kind in ("source", "requirements"):
            data = verdict()
            if kind == "source":
                data["requirements"][0]["source_ids"] = ["invented"]
            else:
                data["requirements"] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ar.validate_review(data, [chunk()])

    def test_invalid_optional_excerpt_is_discarded_not_used_as_fallback(self):
        data = verdict()
        data["supported_excerpts"][0]["quote"] = "Invented policy"
        checked = ar.validate_review(data, [chunk()])
        self.assertEqual("accept", checked["action"])
        self.assertEqual([], checked["supported_excerpts"])
        self.assertEqual(1, checked["discarded_excerpt_count"])

    def test_reviewer_reported_citation_issue_forces_revision(self):
        data = verdict()
        data["citation_issues"] = ["A policy claim is uncited."]
        checked = ar.validate_review(data, [chunk()])
        self.assertEqual("revise", checked["action"])

    def test_structural_citation_guard_rejects_short_and_altered_ids(self):
        chunks = [chunk("Manual_V1.pdf::0029")]
        answer = ("Valid [Manual_V1.pdf::0029], short 【0029】, "
                  "line-style 【0029†L9-L12】, altered [Manual_V1::0029], footnote [1].")
        issues = ar.structural_citation_issues(answer, chunks)
        self.assertEqual(3, len(issues))
        data = verdict()
        data["requirements"][0]["source_ids"] = ["Manual_V1.pdf::0029"]
        data["supported_excerpts"] = []
        checked = ar.validate_review(data, chunks, answer)
        self.assertEqual("revise", checked["action"])
        self.assertEqual(3, len(checked["citation_issues"]))

    def test_style_requirement_does_not_request_policy_search(self):
        data = verdict(evidence="not_needed", answer="missing")
        self.assertEqual("revise", data["action"])

    def test_runtime_prompt_never_uses_answer_key(self):
        state = {"question": QUESTION, "resolved_question": "rewritten", "scope": SCOPE,
                 "context_chunks": [chunk()], "answer": "draft",
                 "required_claims": ["SECRET_ANSWER_KEY"], "evidence_groups": ["SECRET_GROUP"]}
        with patch.object(ar, "_ask", return_value=verdict()) as ask:
            ar.review_draft(state)
        prompt = ask.call_args.args[0]
        self.assertIn(QUESTION, prompt)
        self.assertIn("rewritten", prompt)
        self.assertNotIn("SECRET_ANSWER_KEY", prompt)
        self.assertNotIn("SECRET_GROUP", prompt)
        self.assertEqual(ar.PROMPT, ask.call_args.kwargs["system_prompt"])
        self.assertNotIn(QUESTION, ask.call_args.kwargs["system_prompt"])

    def test_runtime_rubric_preserves_atomic_clauses_and_audits_citations(self):
        prompt = " ".join(ar.PROMPT.split())
        self.assertIn("each independently answerable obligation", prompt)
        self.assertIn("possible distractor", prompt)
        self.assertIn("Do not combine policy", prompt)
        self.assertIn('"citation_issues"', prompt)
        self.assertIn("exact, complete chunk ID", prompt)


class TelemetryTests(unittest.TestCase):
    def node(self, events):
        return {"node": "test", "elapsed_ms": 123, "events": events}

    def test_usage_comes_from_provider_response_not_text_length(self):
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))
        with t.capture() as events:
            returned = t.tracked_call("groq", "generate", "m", lambda **kw: response, model="m")
        self.assertIs(response, returned)
        self.assertEqual(120, events[0]["total_tokens"])
        self.assertEqual(100, events[0]["input_tokens"])
        self.assertEqual("ok", events[0]["status"])

    def test_cache_hit_is_zero_new_api_calls(self):
        with t.capture() as events:
            t.cache_hit("groq", "generate", "m")
        result = t.summarize([self.node(events)], {"models": {}})
        self.assertEqual(0, result["api_calls"])
        self.assertEqual(1, result["cache_hits"])
        self.assertEqual(0, result["estimated_api_cost_usd"])

    def test_unknown_prices_or_usage_are_not_zero_cost(self):
        with t.capture() as events:
            t.tracked_call("voyage", "embed", "m", lambda: SimpleNamespace(total_tokens=17))
        result = t.summarize([self.node(events)], {"models": {}})
        self.assertIsNone(result["estimated_api_cost_usd"])
        self.assertEqual(1, result["calls_without_cost_estimate"])
        self.assertEqual(17, result["known_total_tokens"])

    def test_known_prices_and_missing_prices_remain_distinct(self):
        prices = {"as_of": "test rates, NOT real pricing", "models": {"groq:m": {"input_per_million": 1, "output_per_million": 2}}}
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))
        with t.capture() as events:
            t.tracked_call("groq", "generate", "m", lambda: response)
        summary = t.summarize([self.node(events)], prices)
        self.assertAlmostEqual(0.00014, summary["estimated_api_cost_usd"])
        events.append({**events[0], "model": "unknown"})
        summary = t.summarize([self.node(events)], prices)
        self.assertIsNone(summary["estimated_api_cost_usd"])
        self.assertAlmostEqual(0.00014, summary["known_api_cost_subtotal_usd"])

    def test_nested_captures_do_not_mix_requests(self):
        with t.capture() as first:
            t.cache_hit("groq", "first", "m")
            with t.capture() as second:
                t.cache_hit("groq", "second", "m")
            t.cache_hit("groq", "first-again", "m")
        self.assertEqual(2, len(first))
        self.assertEqual(1, len(second))

    def test_provider_error_is_observed_without_logging_secret_error_body(self):
        with t.capture() as events, self.assertRaises(TimeoutError):
            t.tracked_call("groq", "generate", "m", lambda: (_ for _ in ()).throw(TimeoutError("secret body")))
        self.assertEqual("error", events[0]["status"])
        self.assertEqual("TimeoutError", events[0]["error_type"])
        self.assertNotIn("secret body", str(events))

    def test_rate_limit_logs_only_category_and_allowlisted_headers(self):
        RateLimitError = type("RateLimitError", (Exception,), {})
        error = RateLimitError("secret URL and payload")
        error.response = SimpleNamespace(headers={
            "x-ratelimit-limit-tokens": "8000", "retry-after": "12",
            "authorization": "must-not-log", "set-cookie": "must-not-log",
        })
        error.body = {"error": {"message": "Rate limit reached for tokens per minute",
                                "type": "tokens", "code": "rate_limit_exceeded",
                                "secret": "must-not-log"}}
        with t.capture() as events, self.assertRaises(RateLimitError):
            t.tracked_call("groq", "review", "m", lambda: (_ for _ in ()).throw(error))
        details = events[0]["rate_limit"]
        self.assertEqual("tokens_per_minute", details["category"])
        self.assertEqual({"x-ratelimit-limit-tokens": "8000", "retry-after": "12"}, details["headers"])
        self.assertNotIn("secret", str(events).lower())

    def test_rate_limit_extracts_only_labelled_numeric_limit_and_request(self):
        RateLimitError = type("RateLimitError", (Exception,), {})
        error = RateLimitError("not persisted")
        error.response = SimpleNamespace(headers={})
        error.body = {"error": {"message": "TPM limit 8,000, Requested 9,234"}}
        details = t.rate_limit_details(error)
        self.assertEqual(8000, details["reported_limit"])
        self.assertEqual(9234, details["reported_requested"])

    def test_waits_and_local_tools_are_not_api_calls(self):
        @t.local_tool("lookup")
        def lookup():
            t.wait_event("voyage", 2, "pacing")
        with t.capture() as events:
            lookup()
        summary = t.summarize([self.node(events)], {"models": {}})
        self.assertEqual(2000, summary["wait_ms"])
        self.assertEqual(0, summary["api_calls"])
        self.assertEqual(1, summary["local_tool_calls"])

    def test_node_wrapper_records_nested_provider_call(self):
        def node(state):
            t.tracked_call("voyage", "embed", "m", lambda: SimpleNamespace(total_tokens=17))
            return {"answer": "ok"}
        result = t.observed_node("test", node)({})
        self.assertEqual(17, result["node_events"][0]["events"][0]["total_tokens"])
        self.assertGreaterEqual(result["node_events"][0]["elapsed_ms"], 0)

    def test_qwen_output_budget_accounts_for_prompt_size(self):
        short = gr.output_budget("x" * 3500, 2800)
        long = gr.output_budget("x" * 18000, 2800)
        self.assertEqual(2800, short)
        self.assertLess(long, short)
        self.assertLessEqual(gr.estimate_prompt_tokens("x" * 18000) + long,
                             gr.TPM - gr.SAFETY_TOKENS)

    def test_qwen_reservation_reconciles_to_provider_usage(self):
        gr.reset()
        reservation = gr.reserve("m", "short prompt", 100)
        gr.reconcile("m", reservation, 17)
        self.assertEqual(17, gr._reservations["m"][0]["tokens"])
        gr.reset()

    def test_all_qwen_consumers_share_pacing_before_provider_calls(self):
        import json
        gr.reset()
        self.addCleanup(gr.reset)
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        messages = [{"role": "user", "content": "x" * 140}]
        requested = gr.estimate_prompt_tokens(json.dumps(messages)) + 50
        response = SimpleNamespace(usage=SimpleNamespace(total_tokens=requested))
        with patch.object(gr, "TPM", requested + 20), patch.object(gr, "SAFETY_TOKENS", 0), \
             patch.object(gr.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(gr.time, "sleep", side_effect=sleep), t.capture() as events:
            for operation in ("guardrail_screen", "decompose", "runtime_review"):
                t.tracked_call("groq", operation, "qwen/test", lambda **kwargs: response,
                               messages=messages, max_tokens=50)
        self.assertEqual(3, sum(e["kind"] == "api_call" for e in events))
        self.assertEqual(2, sum(e["kind"] == "wait" for e in events))
        self.assertGreaterEqual(clock[0], 120)

    def test_guard_capacity_wait_is_bounded(self):
        gr.reset()
        self.addCleanup(gr.reset)
        with patch.object(gr, "TPM", 150), patch.object(gr, "SAFETY_TOKENS", 0), \
             patch.object(gr.time, "monotonic", return_value=0), \
             patch.object(gr.time, "sleep") as sleep:
            gr.reserve("qwen/test", "short", 100)
            with self.assertRaises(gr.CapacityWaitTimeout):
                gr.reserve("qwen/test", "short", 100, max_wait_seconds=1)
            sleep.assert_not_called()

    def test_qwen_reconciliation_does_not_undercount_actual_usage(self):
        gr.reset()
        self.addCleanup(gr.reset)
        reservation = gr.reserve("qwen/test", "short", 100)
        gr.reconcile("qwen/test", reservation, 999)
        self.assertEqual(999, gr._reservations["qwen/test"][0]["tokens"])

    def test_retry_runner_copies_measured_usage_without_inventing_unknowns(self):
        usage = {"api_calls": 2, "known_total_tokens": 140, "calls_without_token_usage": 0,
                 "estimated_api_cost_usd": None, "cache_hits": 3}
        attempt = build_attempt({"usage_summary": usage}, 10, "completed")
        self.assertEqual(2, attempt["usage"]["provider_calls"])
        self.assertEqual(140, attempt["usage"]["tokens"])
        self.assertEqual(3, attempt["usage"]["cache_hits"])
        self.assertIsNone(attempt["usage"]["cost_usd"])
        self.assertIsNone(attempt["usage"]["transport_retries"])
        unknown = build_attempt({}, 10, "error")
        self.assertIsNone(unknown["usage"]["provider_calls"])
        usage["calls_without_token_usage"] = 1
        partial = build_attempt({"usage_summary": usage}, 10, "completed")
        self.assertIsNone(partial["usage"]["tokens"])


if __name__ == "__main__":
    unittest.main()
