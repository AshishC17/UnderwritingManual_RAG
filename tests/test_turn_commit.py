"""Block C, offline half: evidence hashing and the stale-evidence fault path.

No network, no database -- state_store.commit_turn's idempotency against real
Postgres is covered separately by scripts/verify_recovery.py, the same split
verify_state_restart.py already uses for Block A.
"""
from __future__ import annotations

import unittest

from src.graph.pipeline import _evidence_fault
from src.store.state_store import evidence_hash

SCOPE = {"status": "resolved", "lender_id": "novacred", "document_version": "V1"}


def chunk(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "lender_id": "novacred", "document_version": "V1"}


class EvidenceHashTests(unittest.TestCase):
    def test_deterministic_for_the_same_ids(self):
        self.assertEqual(evidence_hash(["a", "b"]), evidence_hash(["a", "b"]))

    def test_order_independent(self):
        self.assertEqual(evidence_hash(["a", "b"]), evidence_hash(["b", "a"]))

    def test_different_ids_hash_differently(self):
        self.assertNotEqual(evidence_hash(["a", "b"]), evidence_hash(["a", "c"]))

    def test_accepts_a_generator_not_just_a_list(self):
        self.assertEqual(evidence_hash(["a", "b"]), evidence_hash(x for x in ["a", "b"]))


class StaleEvidenceFaultTests(unittest.TestCase):
    def test_no_expected_hash_means_no_staleness_check(self):
        chunks = [chunk("x::1")]
        self.assertIsNone(_evidence_fault({"scope": SCOPE}, chunks))

    def test_matching_hash_passes(self):
        chunks = [chunk("x::1"), chunk("x::2")]
        expected = evidence_hash(c["chunk_id"] for c in chunks)
        self.assertIsNone(_evidence_fault({"scope": SCOPE}, chunks, expected_evidence_hash=expected))

    def test_mismatched_hash_is_stale_evidence(self):
        chunks = [chunk("x::1"), chunk("x::2")]
        stale_expected = evidence_hash(["x::1", "x::999"])
        self.assertEqual(
            "stale_evidence",
            _evidence_fault({"scope": SCOPE}, chunks, expected_evidence_hash=stale_expected),
        )

    def test_wrong_scope_is_reported_before_staleness_is_even_checked(self):
        chunks = [chunk("x::1")]
        chunks[0]["document_version"] = "V0"  # scope says V1
        expected = evidence_hash(c["chunk_id"] for c in chunks)  # would otherwise match
        self.assertEqual(
            "wrong_scope",
            _evidence_fault({"scope": SCOPE}, chunks, expected_evidence_hash=expected),
        )

    def test_empty_context_is_reported_before_staleness_is_even_checked(self):
        self.assertEqual(
            "empty_context",
            _evidence_fault({"scope": SCOPE}, [], expected_evidence_hash="anything"),
        )


if __name__ == "__main__":
    unittest.main()
