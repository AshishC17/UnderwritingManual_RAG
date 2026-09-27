"""Offline tests: local Qdrant, temporary artifacts, no model/API calls."""
import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient
from src.store import index_builds as ib, qdrant_store as qs
from src.store.index_context import bind, collection_name, corpus_path
from src.eval.excerpts import evidence_result, contains_anchor
from scripts.run_index_eval import select_context


class ExcerptV2Tests(unittest.TestCase):
    def setUp(self):
        self.item = {"document": "manual.pdf", "kind": "table_row",
                     "text": "| Credit Freeze | RFAI | 105 |", "context": "Table 3: Prescreen",
                     "required_headers": ["| Rule | Action | Code |"],
                     "source_scope": {"lender_id": "nc", "document_version": "V1"},
                     "authoring_chunk_id": "old-id"}
        self.chunk = {"chunk_id": "new-id", "source_doc": "manual.pdf", "lender_id": "nc", "document_version": "V1",
                      "text": "Table 3: Prescreen\n| Rule | Action | Code |\n| --- | --- | --- |\n| Credit Freeze | RFAI | 105 |"}

    def test_renumbering_does_not_require_relabelling(self):
        self.assertEqual(evidence_result(self.item, [self.chunk])["chunk_ids"], ["new-id"])

    def test_missing_header_fails_even_when_row_exists(self):
        self.chunk["text"] = self.chunk["text"].replace("| Rule | Action | Code |", "")
        self.assertEqual(evidence_result(self.item, [self.chunk])["status"], "wrong_context")

    def test_wrong_scope_cannot_satisfy(self):
        self.chunk["document_version"] = "V0"
        self.assertFalse(evidence_result(self.item, [self.chunk])["resolved"])

    def test_wrong_table_in_same_chunk_cannot_borrow_other_header(self):
        self.chunk["text"] = "Table 3: Prescreen\n| Rule | Action | Code |\n| --- | --- | --- |\n| Other | NOAA | 101 |\nTable 4: Full underwriting\n| Rule | Action | Code |\n| --- | --- | --- |\n| Credit Freeze | RFAI | 105 |"
        self.assertFalse(evidence_result(self.item, [self.chunk])["resolved"])

    def test_flattened_row_does_not_count_as_intact_table(self):
        self.chunk["text"] = self.chunk["text"].replace("| Credit Freeze | RFAI | 105 |", "Credit Freeze RFAI 105")
        self.assertFalse(evidence_result(self.item, [self.chunk])["resolved"])

    def test_duplicate_overlap_needs_only_one_witness(self):
        result = evidence_result(self.item, [self.chunk, {**self.chunk, "chunk_id": "other-id"}])
        self.assertEqual(len(result["chunk_ids"]), 1)
        self.assertEqual(len(result["all_matching_chunk_ids"]), 2)

    def test_number_and_negation_changes_are_not_intact(self):
        self.assertFalse(contains_anchor("score 8.00", "score 800"))
        self.assertFalse(contains_anchor("approval is allowed", "approval is not allowed"))

    def test_prose_can_span_new_chunks(self):
        item = {"document": "x", "text": "alpha beta gamma delta epsilon zeta eta theta"}
        chunks = [{"chunk_id": "a", "source_doc": "x", "text": "alpha beta gamma delta"},
                  {"chunk_id": "b", "source_doc": "x", "text": "epsilon zeta eta theta"}]
        self.assertEqual(evidence_result(item, chunks)["status"], "split")

    def test_budget_never_slices_a_chunk_or_exceeds_cap(self):
        with patch("scripts.run_index_eval.n_tokens", return_value=20):
            kept, used = select_context([self.chunk], 10, 1)
        self.assertEqual((kept, used), ([], 0))


class IndexLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "builds"
        source = Path(self.tmp.name) / "manual.pdf"
        source.write_text("synthetic fixture, not a real PDF")
        self.corpus = Path(self.tmp.name) / "corpus.json"
        spec = {"source_doc": "manual.pdf", "path": str(source), "lender_id": "nc", "document_id": "uw",
                "document_version": "V1", "version_status": "current", "effective_from": "2026-01-01T00:00:00Z", "effective_to": None}
        self.corpus.write_text(json.dumps({"chunking_version": "fixture", "documents": [spec]}))
        payload = {k:v for k,v in spec.items() if k != "path"}
        payload.update(chunk_id="manual.pdf::01", text="Credit freeze requires additional information", embedding_model="fixture-model")
        self.points = [{"id": qs.point_id(payload["chunk_id"]), "payload": payload,
                        "vector": {qs.DENSE: [1., 0.], qs.SPARSE: {"indices": [1], "values": [1.]}}}]
        self.client = QdrantClient(":memory:")
        for name in ("a", "b"):
            ib._snapshot(name, self.points, self.corpus, {"model": "fixture-model", "dims": 2},
                         {"kind": "test_fixture"}, [], root=self.root)
            ib.install_candidate(self.client, name, self.root)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def release(self, name, expected=None, **kwargs):
        return ib.release(self.client, name, expected_collection=expected, reason="test", root=self.root, **kwargs)

    def test_validate_candidate_and_never_overwrite(self):
        self.assertTrue(ib.validate_collection(self.client, "a", self.root)["integrity_passed"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            ib.install_candidate(self.client, "a", self.root)

    def test_artifact_tampering_is_detected(self):
        (self.root / "a/chunks.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "integrity"):
            ib.binding_for("a", self.root)

    def test_alias_promotion_rollback_and_stale_preconditions(self):
        self.release("a", bootstrap=True)
        pinned_a = ib.resolve_active(self.client, root=self.root)
        with self.assertRaisesRegex(ValueError, "evaluation"):
            self.release("b", "uw_idx_a")
        report = Path(self.tmp.name) / "report.json"
        report.write_text(json.dumps({"promotion_eligible": True,
            "active_manifest_sha256": pinned_a["manifest_sha256"],
            "candidate_manifest_sha256": ib.binding_for("b", self.root)["manifest_sha256"]}))
        self.release("b", "uw_idx_a", evaluation=str(report))
        with bind(pinned_a):
            self.assertEqual(collection_name(), "uw_idx_a")
            self.assertEqual(len(qs.get_by_chunk_ids(self.client, ["manual.pdf::01"])), 1)
        self.assertEqual(ib.resolve_active(self.client, root=self.root)["build_id"], "b")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.release("a", "uw_idx_a", rollback=True)
        self.release("a", "uw_idx_b", rollback=True)
        self.assertEqual(ib.resolve_active(self.client, root=self.root)["build_id"], "a")
        events = [json.loads(line) for line in (self.root / "releases.jsonl").read_text().splitlines()]
        self.assertEqual([e["event"] for e in events], ["intent", "completed"]*3)

    def test_unreleased_build_cannot_be_rollback_target(self):
        self.release("a", bootstrap=True)
        with self.assertRaisesRegex(ValueError, "never released"):
            self.release("b", "uw_idx_a", rollback=True)

    def test_context_isolation_and_exception_cleanup(self):
        def run(name):
            binding = ib.binding_for(name, self.root)
            with bind(binding):
                self.assertEqual(corpus_path("config/corpus_manifest.json"), (self.root / name / "corpus.json").resolve())
                return collection_name()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            self.assertEqual(list(pool.map(run, ["a", "b"])), ["uw_idx_a", "uw_idx_b"])
        try:
            with bind(ib.binding_for("a", self.root)):
                raise RuntimeError("fixture")
        except RuntimeError:
            pass
        self.assertEqual(collection_name(), "uw_manual")

    def test_pin_rejects_changed_manifest(self):
        binding = ib.binding_for("a", self.root)
        binding["dense"]["dims"] = 3
        with self.assertRaisesRegex(ValueError, "binding changed"):
            ib.validate_binding(binding)

    def test_failed_alias_call_leaves_reconcilable_intent(self):
        with patch.object(self.client, "update_collection_aliases", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.release("a", bootstrap=True)
        events = [json.loads(line) for line in (self.root / "releases.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "unknown_outcome")
        outcome = ib.reconcile(self.client, events[-1]["operation_id"], self.root)
        self.assertEqual(outcome["event"], "not_applied")

    def test_graph_resumes_pinned_build_but_new_turn_selects_fresh(self):
        from src.graph import pipeline as p
        from langgraph.types import Command
        chosen = [ib.binding_for("a", self.root)]
        calls, observed = [], []
        def selector():
            calls.append(chosen[0]["build_id"])
            return chosen[0]
        def contextualize(state):
            return {"behavior": "answer_policy_question"}
        def resolve(state):
            observed.append(collection_name())
            return {"scope": {"status": "needs_clarification", "reason": "fixture",
                               "clarification_question": "Which lender?"}}
        with patch.object(p, "contextualize_node", contextualize), patch.object(p, "resolve_scope_node", resolve):
            graph = p.build_graph(index_selector=selector)
            config = {"configurable": {"thread_id": "pin-test"}}
            first = graph.invoke({"question": "Which policy?"}, config)
            self.assertIn("__interrupt__", first)
            chosen[0] = ib.binding_for("b", self.root)
            resumed = graph.invoke(Command(resume="still ambiguous"), config)
            self.assertEqual(resumed["index_binding"]["build_id"], "a")
            self.assertEqual(calls, ["a"])
            self.assertEqual(observed, ["uw_idx_a", "uw_idx_a"])
            graph.invoke(Command(resume="still ambiguous"), config)
            fresh = graph.invoke({"question": "New policy?"}, config)
            self.assertEqual(fresh["index_binding"]["build_id"], "b")
            self.assertEqual(calls, ["a", "b"])

    def test_rephrase_selects_original_evidence_build(self):
        from src.graph.pipeline import contextualize_node
        state = {"question": "simplify that", "index_binding": ib.binding_for("b", self.root),
                 "history": [{"question": "policy", "answer": "prior", "index_binding": ib.binding_for("a", self.root)}]}
        with patch("src.graph.pipeline.detect_meta_intent", return_value=None), patch(
                "src.graph.pipeline.classify_followup", return_value={"behavior": "transform_previous_answer",
                    "standalone": "policy", "instruction": "simplify"}):
            result = contextualize_node(state)
        self.assertEqual(result["index_binding"]["build_id"], "a")


if __name__ == "__main__":
    unittest.main()
