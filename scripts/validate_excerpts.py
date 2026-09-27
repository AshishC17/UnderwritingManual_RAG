#!/usr/bin/env python3
"""Validate eval/excerpts_v1.json against the dev eval, its sources and a chunk file."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.excerpts import assess, identified_in, tokens  # noqa: E402


def short(doc: str) -> str:
    return doc.replace("_pdf.pdf", "")


def load_chunks(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else raw["chunks"]


def eval_groups(path: Path) -> dict:
    data = json.loads(path.read_text())
    cases = data.get("cases", data)
    return {
        (case["id"], g["name"]): (g["requirement"], tuple(g["any_of_source_refs"]))
        for case in cases
        for g in case.get("evidence_groups") or []
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpts", default="eval/excerpts_v1.json")
    ap.add_argument("--eval", default="eval/retry_eval_v1/dev_eval.json")
    ap.add_argument("--sources", default="eval/retry_eval_v1/sources.json")
    ap.add_argument("--chunks", default=None, help="defaults to the chunk file the excerpts were drafted against")
    ap.add_argument("--candidate", action="store_true", help="map anchors independently of authoring IDs")
    ap.add_argument("--out", help="write candidate mapping report")
    args = ap.parse_args()

    doc = json.loads((ROOT / args.excerpts).read_text())
    excerpts, groups = doc["excerpts"], doc["groups"]
    sources = json.loads((ROOT / args.sources).read_text())["sources"]
    chunk_path = ROOT / (args.chunks or doc["corpus_snapshot"]["chunks_path"])
    chunks = load_chunks(chunk_path)
    if args.candidate:
        from src.eval.excerpts import map_evidence
        from src.store.index_builds import write_once
        report = map_evidence(doc, chunks)
        report["chunks_sha256"] = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
        report["reference_status"] = doc["status"]
        if args.out:
            write_once(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report["all_groups_covered"] else 1
    by_id = {c["chunk_id"]: c for c in chunks}
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    sha = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    if sha != doc["corpus_snapshot"]["chunks_sha256"]:
        warnings.append("chunk file differs from the one the excerpts were drafted against")

    expected = eval_groups(ROOT / args.eval)
    actual = {(g["case"], g["group"]): (g["requirement"], tuple(a["ref"] for a in g["any_of"])) for g in groups}
    for key in sorted(expected.keys() - actual.keys()):
        errors.append(f"group {key[0]}/{key[1]} is in the eval but has no excerpts")
    for key in sorted(actual.keys() - expected.keys()):
        errors.append(f"group {key[0]}/{key[1]} is not in the eval")
    for key in sorted(expected.keys() & actual.keys()):
        if expected[key] != actual[key]:
            errors.append(f"group {key[0]}/{key[1]} differs from the eval (requirement or alternatives)")

    refs_of: dict[str, set[str]] = {}
    for g in groups:
        for alt in g["any_of"]:
            if not alt["excerpts"]:
                errors.append(f"{g['case']}/{g['group']} alternative {alt['ref']} lists no excerpts")
            for eid in alt["excerpts"]:
                if eid not in excerpts:
                    errors.append(f"unknown excerpt id {eid}")
                    continue
                refs_of.setdefault(eid, set()).add(alt["ref"])
                if excerpts[eid]["authoring_chunk_id"] != sources[alt["ref"]]["chunk_id"]:
                    errors.append(f"{eid}: authoring chunk differs from sources.json chunk for {alt['ref']}")
    for eid in sorted(excerpts.keys() - refs_of.keys()):
        warnings.append(f"{eid} is not used by any group")

    rows = []
    for eid, item in excerpts.items():
        text, context = item["text"], item.get("context")
        if not tokens(text) or (context is not None and not tokens(context)):
            errors.append(f"{eid}: no matchable tokens")
            continue
        authoring = by_id.get(item["authoring_chunk_id"])
        if authoring is None:
            errors.append(f"{eid}: authoring chunk {item['authoring_chunk_id']} not in the chunk file")
            continue
        if authoring["source_doc"] != item["document"]:
            errors.append(f"{eid}: document {item['document']} differs from its chunk's {authoring['source_doc']}")
        if item["authoring_chunk_id"] not in identified_in(text, context, [authoring]):
            errors.append(f"{eid}: excerpt with its context is not intact in its authoring chunk")
        own_chunks = [c for c in chunks if c["source_doc"] == item["document"]]
        other_chunks = [c for c in chunks if c["source_doc"] != item["document"]]
        text_only = assess(text, own_chunks).intact_in
        identified = identified_in(text, context, own_chunks)
        elsewhere = sorted({short(by_id[i]["source_doc"]) for i in identified_in(text, context, other_chunks)})
        if len(identified) != 1:
            warnings.append(f"{eid}: found in {len(identified)} chunks of its own document, expected exactly 1")
        if elsewhere:
            warnings.append(f"{eid}: same text and context also appear in {', '.join(elsewhere)}; match on document as well")
        rows.append((eid, len(tokens(text)), len(text_only), len(identified), ", ".join(elsewhere) or "-"))

    for ref in sorted({r for rs in refs_of.values() for r in rs}):
        pieces = [{"chunk_id": eid, "text": excerpts[eid]["text"]} for eid, rs in refs_of.items() if ref in rs]
        if assess(sources[ref]["anchor"], pieces).status != "intact":
            notes.append(f"{ref}: sources.json anchor is not inside any excerpt for it ({sources[ref]['anchor'][:60]!r})")

    print(f"excerpts: {len(excerpts)}  groups: {len(groups)}  status: {doc['status']}")
    print(f"chunk file: {chunk_path.relative_to(ROOT)} ({len(chunks)} chunks)\n")
    print(f"{'excerpt id':44s} {'tok':>4s} {'text in':>8s} {'+context':>9s}  also in other docs")
    for eid, n, text_count, identified_count, other in rows:
        print(f"{eid:44s} {n:4d} {text_count:8d} {identified_count:9d}  {other}")
    for label, items in (("ERROR", errors), ("WARNING", warnings), ("NOTE", notes)):
        for line in items:
            print(f"{label}: {line}")
    print(f"\nerrors: {len(errors)}  warnings: {len(warnings)}  notes: {len(notes)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
