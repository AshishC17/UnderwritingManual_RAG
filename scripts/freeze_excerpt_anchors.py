"""Create a NEW draft reference file, preserving v1. No LLM/API calls.

Authoring IDs remain audit hints only. Candidate evaluation uses source hashes,
scope and text/structure anchors. A human still needs to review the PDF against
these draft excerpts; automatic extraction is NOT independent ground truth.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.store.index_builds import digest, write_once


def freeze(source, chunks_path, corpus_path, out):
    doc = json.loads(Path(source).read_text())
    chunks = {c["chunk_id"]: c for c in json.loads(Path(chunks_path).read_text())}
    corpus = json.loads(Path(corpus_path).read_text())
    doc.update(schema_version="2.0", status="draft_pending_review")
    doc["conventions"] = {"identity": "Source PDF hash + source scope + text/structure anchors; authoring IDs are nonbinding audit hints.",
                          "table_integrity": "Row, caption and declared headers must coexist in a chunk.",
                          "limitations": "Surface matching is not semantic completeness. Human source review pending; do not auto-approve labels."}
    doc["source_sha256"] = {d["source_doc"]: digest((ROOT / d["path"]).read_bytes()) for d in corpus["documents"]}
    for eid, item in doc["excerpts"].items():
        chunk = chunks[item["authoring_chunk_id"]]
        item["source_scope"] = {k: chunk[k] for k in ("lender_id", "document_id", "document_version")}
        item["required_headers"] = []
        if item.get("kind") == "table_row":
            lines = chunk["text"].splitlines()
            headers = [lines[i-1] for i, line in enumerate(lines) if i and
                       re.fullmatch(r"\s*\|[\s:|\-]+\|\s*", line)]
            if len(headers) != 1:
                raise ValueError(f"{eid}: ambiguous/missing markdown header; manual authoring required")
            item["required_headers"] = headers
    write_once(Path(out), doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="eval/excerpts_v1.json")
    p.add_argument("--chunks", default="data/processed/chunks_v2.json")
    p.add_argument("--corpus", default="config/corpus_manifest.json")
    p.add_argument("--out", default="eval/excerpts_v2.json")
    a = p.parse_args()
    d = freeze(a.source, a.chunks, a.corpus, a.out)
    print(f"Wrote {a.out}: {len(d['excerpts'])} draft anchors; human review still required.")
