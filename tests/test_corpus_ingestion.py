from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.run_embed import embed_by_document
from src.ingest.chunker import chunk
from src.ingest.manifest import assign_chunk_revisions, load_manifest
from src.ingest.parser import Block, Document
from src.store.qdrant_store import scope_filter


def _document(source: str, text: str) -> Document:
    return Document(
        source=source,
        blocks=[
            Block(kind="heading", page=1, text="ELIGIBILITY GATE", level=1),
            Block(kind="heading", page=1, text="INCOME RULE", level=3),
            Block(kind="prose", page=1, text=text),
        ],
    )


def _metadata(version: str, order: int) -> dict:
    return {
        "lender_id": "test_lender",
        "lender_name": "Test Lender",
        "product_id": "test_product",
        "product_name": "Test Product",
        "document_family": "underwriting_manual",
        "document_id": "test_product_uw",
        "document_version": version,
        "version_order": order,
        "effective_from": "2026-01-01T00:00:00Z",
        "effective_to": None,
        "version_status": "current",
        "supersedes_version": "V0" if order else None,
        "authority_level": "authoritative_underwriting_policy",
        "chunking_version": "structure-v2",
    }


class CorpusMetadataTests(unittest.TestCase):
    def test_manifest_has_two_lenders_and_two_versions_each(self):
        specs = load_manifest("config/corpus_manifest.json", require_files=False)

        self.assertEqual(4, len(specs))
        self.assertEqual({"novacred", "lumentrail"}, {s.lender_id for s in specs})
        for lender in {s.lender_id for s in specs}:
            family = [s for s in specs if s.lender_id == lender]
            self.assertEqual({"V0", "V1"}, {s.document_version for s in family})
            self.assertEqual(1, sum(s.version_status == "current" for s in family))

    def test_positional_id_is_preserved_and_lineage_is_version_independent(self):
        with patch("src.ingest.chunker.n_tokens", side_effect=lambda text: len(text.split())):
            v0 = chunk(
                _document("Manual_V0.pdf", "Minimum income is 2,200 dollars."),
                _metadata("V0", 0),
            )[0]
            v1 = chunk(
                _document("Manual_V1.pdf", "Minimum income is 2,350 dollars."),
                _metadata("V1", 1),
            )[0]

        self.assertEqual("Manual_V0.pdf::0000", v0.chunk_id)
        self.assertEqual("Manual_V1.pdf::0000", v1.chunk_id)
        self.assertEqual(v0.lineage_id, v1.lineage_id)
        self.assertNotEqual(v0.content_hash, v1.content_hash)
        self.assertEqual("eligibility.gate", v0.section_key)
        self.assertEqual("income.rule", v0.subsection_key)

        assign_chunk_revisions([v1, v0])
        self.assertEqual(1, v0.chunk_revision)
        self.assertEqual(2, v1.chunk_revision)

    def test_unchanged_cross_version_content_keeps_revision(self):
        text = "The same evidence rule applies."
        with patch("src.ingest.chunker.n_tokens", side_effect=lambda value: len(value.split())):
            v0 = chunk(_document("Manual_V0.pdf", text), _metadata("V0", 0))[0]
            v1 = chunk(_document("Manual_V1.pdf", text), _metadata("V1", 1))[0]

        assign_chunk_revisions([v0, v1])

        self.assertEqual(1, v0.chunk_revision)
        self.assertEqual(1, v1.chunk_revision)


class ContextualEmbeddingGroupingTests(unittest.TestCase):
    def setUp(self):
        p = patch("src.guardrails.checks._semantic_call", return_value='{"decision":"allow","reason":"ordinary_content"}')
        p.start()
        self.addCleanup(p.stop)

    def test_documents_are_embedded_separately_and_output_order_is_restored(self):
        chunks = [
            {"source_doc": "a.pdf", "text": "a1"},
            {"source_doc": "b.pdf", "text": "b1"},
            {"source_doc": "a.pdf", "text": "a2"},
        ]

        def fake_embed(texts, *, model, dims):
            base = 10 if texts[0].startswith("a") else 20
            return [[base + index] for index, _ in enumerate(texts)]

        with patch("scripts.run_embed.embed_document", side_effect=fake_embed) as mocked:
            vectors = embed_by_document(chunks, model="contextual", dims=1)

        self.assertEqual([[10], [20], [11]], vectors)
        self.assertEqual(2, mocked.call_count)
        self.assertEqual(["a1", "a2"], mocked.call_args_list[0].args[0])
        self.assertEqual(["b1"], mocked.call_args_list[1].args[0])


class ScopeFilterTests(unittest.TestCase):
    def test_filter_contains_lender_and_effective_interval(self):
        payload = scope_filter(
            lender_id="novacred", as_of="2026-06-30T12:00:00Z"
        ).model_dump(exclude_none=True)

        self.assertEqual("novacred", payload["must"][0]["match"]["value"])
        self.assertEqual("effective_from", payload["must"][1]["key"])
        self.assertEqual(3, len(payload["must"][2]["should"]))

    def test_current_and_as_of_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            scope_filter(current_only=True, as_of="2026-06-30T12:00:00Z")


if __name__ == "__main__":
    unittest.main()
