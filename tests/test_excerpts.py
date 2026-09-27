"""Excerpt matching classifies evidence as intact, split, partial or missing whatever the chunker."""
from __future__ import annotations

import unittest

from src.eval.excerpts import assess, assess_by_document, identified_in, tokens

EXCERPT = "The exception requires a resolved administrative duplicate older than forty eight months"


def chunk(chunk_id: str, text: str, doc: str = "doc.pdf") -> dict:
    return {"chunk_id": chunk_id, "text": text, "source_doc": doc}


class TokenTests(unittest.TestCase):
    def test_case_markdown_and_punctuation_do_not_matter(self):
        self.assertEqual(
            ["blocked", "ssn", "1", "noaa", "101"],
            tokens("| Blocked SSN [1] | NOAA | 101 |"),
        )

    def test_comparison_symbols_are_kept(self):
        self.assertNotEqual(tokens("score ≥ 700"), tokens("score 700"))


class AssessTests(unittest.TestCase):
    def test_excerpt_inside_one_chunk_is_intact(self):
        result = assess(EXCERPT, [chunk("a", f"Note. {EXCERPT}. More text follows."), chunk("b", "unrelated words here")])
        self.assertEqual("intact", result.status)
        self.assertEqual(["a"], result.intact_in)
        self.assertEqual(1.0, result.coverage)

    def test_excerpt_in_an_overlap_is_intact_in_both_chunks(self):
        left = chunk("a", "alpha beta gamma delta epsilon zeta eta theta")
        right = chunk("b", "epsilon zeta eta theta iota kappa lambda mu")
        result = assess("epsilon zeta eta theta", [left, right])
        self.assertEqual("intact", result.status)
        self.assertEqual(["a", "b"], result.intact_in)

    def test_excerpt_cut_across_two_chunks_is_split(self):
        first = chunk("a", "Policy note. The exception requires a resolved administrative")
        second = chunk("b", "duplicate older than forty eight months. Other text")
        result = assess(EXCERPT, [first, second])
        self.assertEqual("split", result.status)
        self.assertEqual([], result.intact_in)
        self.assertEqual(1.0, result.coverage)

    def test_only_part_present_is_partial(self):
        result = assess(EXCERPT, [chunk("a", "The exception requires a resolved administrative and nothing else")])
        self.assertEqual("partial", result.status)
        self.assertEqual(0.5, result.coverage)

    def test_absent_excerpt_is_missing(self):
        result = assess(EXCERPT, [chunk("a", "completely different wording about lending limits")])
        self.assertEqual("missing", result.status)
        self.assertEqual(0.0, result.coverage)

    def test_excerpt_without_matchable_tokens_is_rejected(self):
        with self.assertRaises(ValueError):
            assess("| --- |", [chunk("a", "text")])


class ContextTests(unittest.TestCase):
    ROW = "| Blocked SSN [1] | NovaShield Identity Blocklist | NOAA | 101 |"

    def sibling_tables(self):
        return [
            chunk("t3", f"Table 3: Prescreen Prequalification Rules\n{self.ROW}"),
            chunk("t5", f"Table 5: Non-Prescreen Prequalification Rules\n{self.ROW}"),
        ]

    def test_same_row_in_sibling_tables_matches_both_without_context(self):
        self.assertEqual(["t3", "t5"], identified_in(self.ROW, None, self.sibling_tables()))

    def test_caption_context_picks_out_the_right_table(self):
        self.assertEqual(
            ["t3"], identified_in(self.ROW, "Table 3: Prescreen Prequalification Rules", self.sibling_tables())
        )

    def test_context_in_a_different_chunk_does_not_count(self):
        chunks = [chunk("a", self.ROW), chunk("b", "Table 3: Prescreen Prequalification Rules")]
        self.assertEqual([], identified_in(self.ROW, "Table 3: Prescreen Prequalification Rules", chunks))


class ByDocumentTests(unittest.TestCase):
    def test_same_text_in_two_documents_is_reported_for_each(self):
        chunks = [chunk("v0::1", f"{EXCERPT}.", "v0.pdf"), chunk("v1::1", f"{EXCERPT}.", "v1.pdf")]
        result = assess_by_document(EXCERPT, chunks)
        self.assertEqual({"v0.pdf": "intact", "v1.pdf": "intact"}, {d: a.status for d, a in result.items()})


if __name__ == "__main__":
    unittest.main()
