from __future__ import annotations

import unittest

from src.eval.comparison_metrics import aggregate_comparison, score_comparison_execution


class ComparisonMetricTests(unittest.TestCase):
    def setUp(self):
        self.keys = ["novacred::novacred_flex_uw::V1", "lumentrail::lumentrail_pathway_uw::V1"]
        self.scopes = [
            {"lender_id": "novacred", "document_id": "novacred_flex_uw", "document_version": "V1"},
            {"lender_id": "lumentrail", "document_id": "lumentrail_pathway_uw", "document_version": "V1"},
        ]
        self.ids = ["NovaCred_UW_V1_pdf.pdf::0007", "LumenTrail_UW_V1_pdf.pdf::0006"]
        self.case = {"expected": {
            "behavior": "compare_policy", "outcome": "answer", "scopes": self.scopes,
            "retry_scope_keys": [],
            "evidence_groups": [
                {"name": "n", "scope_key": self.keys[0], "any_of_chunk_ids": [self.ids[0]]},
                {"name": "l", "scope_key": self.keys[1], "any_of_chunk_ids": [self.ids[1]]},
            ],
        }}

    def chunk(self, index):
        scope = self.scopes[index]
        return {**scope, "chunk_id": self.ids[index], "text": "policy"}

    def test_scores_bilateral_coverage_and_scope_isolation(self):
        final = {
            "behavior": "compare_policy", "comparison_scopes": self.scopes,
            "comparison_context": {
                self.keys[0]: [self.chunk(0)], self.keys[1]: [self.chunk(1)],
            },
            "answer": f"A [{self.ids[0]}]. B [{self.ids[1]}].",
            "merge_lost_chunk_ids": [], "usage_summary": {"request_elapsed_ms": 100},
        }
        score = score_comparison_execution(self.case, final)
        self.assertEqual(1, score["bilateral_full_coverage"])
        self.assertEqual(1, score["both_scope_accuracy"])
        self.assertEqual(0.0, score["wrong_scope_chunk_rate"])
        self.assertEqual(1.0, score["citation_scope_coverage"])
        self.assertTrue(score["orchestration_pass"])

    def test_average_recall_does_not_hide_a_missing_lender(self):
        final = {
            "behavior": "compare_policy", "comparison_scopes": self.scopes,
            "comparison_context": {self.keys[0]: [self.chunk(0)], self.keys[1]: []},
            "answer": "Only one side.", "merge_lost_chunk_ids": [],
        }
        score = score_comparison_execution(self.case, final)
        self.assertEqual(0.5, score["evidence_group_recall"])
        self.assertEqual(0, score["bilateral_full_coverage"])

    def test_guard_failure_is_not_counted_as_a_completed_answer(self):
        score = score_comparison_execution(
            self.case,
            {"guard_reason": "semantic_guard_unavailable", "answer": "Please try again."},
        )
        self.assertFalse(score["outcome_pass"])
        self.assertFalse(score["orchestration_pass"])

    def test_safe_fallback_is_not_counted_as_a_successful_answer(self):
        final = {
            "behavior": "compare_policy",
            "comparison_scopes": self.scopes,
            "comparison_context": {
                self.keys[0]: [self.chunk(0)], self.keys[1]: [self.chunk(1)],
            },
            "answer": "I couldn't verify a complete answer.",
            "final_status": "review_error",
            "merge_lost_chunk_ids": [],
        }
        score = score_comparison_execution(self.case, final)
        self.assertEqual("review_error", score["final_status"])
        self.assertTrue(score["behavior_pass"])
        self.assertFalse(score["outcome_pass"])
        self.assertFalse(score["orchestration_pass"])

    def test_guard_diagnostics_preserve_retrieval_metrics_without_passing_outcome(self):
        final = {
            "behavior": "compare_policy",
            "guard_reason": "invalid_citation",
            "comparison_scopes": self.scopes,
            "comparison_context": {
                self.keys[0]: [self.chunk(0)], self.keys[1]: [self.chunk(1)],
            },
            "citation_tokens": [self.ids[0], "LumenTrail_UW_V1_pdf::0006"],
            "merge_lost_chunk_ids": [],
        }
        score = score_comparison_execution(self.case, final)
        self.assertEqual(1.0, score["evidence_group_recall"])
        self.assertEqual(1, score["bilateral_full_coverage"])
        self.assertEqual(0.5, score["citation_validity"])
        self.assertEqual(0.5, score["citation_scope_coverage"])
        self.assertFalse(score["outcome_pass"])
        self.assertFalse(score["orchestration_pass"])

    def test_aggregate_keeps_metric_denominators(self):
        summary = aggregate_comparison([
            {"behavior_pass": True, "outcome_pass": True, "both_scope_accuracy": 1,
             "evidence_group_recall": 1.0, "bilateral_full_coverage": 1,
             "wrong_scope_chunk_rate": 0.0, "citation_validity": None,
             "citation_scope_coverage": None, "retry_side_accuracy": None,
             "merge_preservation_pass": True, "orchestration_pass": True,
             "latency_ms": 10, "provider_calls": None, "input_tokens": None, "output_tokens": None}
        ])
        self.assertEqual(0, summary["denominators"]["citation_validity"])
        self.assertIsNone(summary["citation_validity"])


if __name__ == "__main__":
    unittest.main()
