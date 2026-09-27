"""Offline dataset validation; no models, Qdrant, tracing, or file writes.

Run with the project's .venv Python. Validates the deliberately small JSON
Schema subset used by schema.json, cross-references, split isolation, corpus
fingerprints, and source anchors in original PDFs. This does not score a model
or prove semantic correctness of the authored answers. Visual anchors were
inspected during authoring and cannot be revalidated by PDF text extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((HERE / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_schema(value, schema, root, path="$", *, check_keywords=True):
    """Validate exactly the schema keywords used by this suite, fail on others."""
    supported = {
        "$ref", "$schema", "title", "description", "definitions", "type", "properties",
        "required", "additionalProperties", "items", "minItems", "enum",
    }
    if check_keywords:
        require(not set(schema) - supported, f"Unsupported schema keywords at {path}")
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return check_schema(value, target, root, path)
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
    }
    require(not types or any(matches.get(t, False) for t in types), f"Wrong type at {path}")
    if "enum" in schema:
        require(value in schema["enum"], f"Unexpected value at {path}: {value!r}")
    if isinstance(value, dict):
        require(set(schema.get("required", [])) <= set(value), f"Missing keys at {path}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            subschema = properties.get(key, additional)
            require(subschema is not False, f"Unknown key at {path}.{key}")
            if isinstance(subschema, dict):
                check_schema(item, subschema, root, f"{path}.{key}")
    if isinstance(value, list):
        require(len(value) >= schema.get("minItems", 0), f"Too few items at {path}")
        for i, item in enumerate(value):
            check_schema(item, schema.get("items", {}), root, f"{path}[{i}]")


def compact(text):
    # Normalize layout whitespace and typographic hyphens, not words or numbers.
    return re.sub(r"\s+", "", text.replace("\u00ad", "").translate(str.maketrans("–—‑", "---")))


def validate_sources(catalog):
    from pypdf import PdfReader

    snapshot = catalog["corpus_snapshot"]
    paths = {
        snapshot["manifest_path"]: snapshot["manifest_sha256"],
        snapshot["chunks_path"]: snapshot["chunks_sha256"],
        **{doc["path"]: doc["sha256"] for doc in snapshot["documents"]},
    }
    for path, digest in paths.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        require(actual == digest, f"Snapshot changed: {path}. Review mappings; do not silently relabel.")
    chunks = json.loads((ROOT / snapshot["chunks_path"]).read_text())
    require(len(chunks) == snapshot["chunk_count"], "Chunk count mismatch")
    by_id = {c["chunk_id"]: c for c in chunks}
    require(len(by_id) == len(chunks), "Duplicate corpus chunk IDs")
    readers = {}
    for source_id, source in catalog["sources"].items():
        require(source["chunk_id"] in by_id, f"Unknown chunk for {source_id}")
        chunk = by_id[source["chunk_id"]]
        for key in ["content_hash", "source_doc", "lender_id", "document_version", "pages"]:
            require(source[key] == chunk[key], f"Source/chunk mismatch: {source_id}.{key}")
        doc = source["source_doc"]
        if doc not in readers:
            readers[doc] = PdfReader(ROOT / "data/raw" / doc)
        reader = readers[doc]
        require(all(1 <= p <= len(reader.pages) for p in source["pages"]), f"Bad page: {source_id}")
        if source["anchor_kind"] == "text":
            text = " ".join(reader.pages[p - 1].extract_text() for p in source["pages"])
            require(compact(source["source_anchor"]) in compact(text), f"PDF anchor mismatch: {source_id}")
        else:
            require(source["anchor_kind"] == "visual", f"Unknown anchor kind: {source_id}")


def validate_splits(datasets, catalog, manifest):
    sources = catalog["sources"]
    all_ids, family_splits, fixture_splits = set(), {}, {}
    for split, data in datasets.items():
        require(data["split"] == split, "Dataset split label mismatch")
        ids = [case["id"] for case in data["cases"]]
        require(ids == manifest[split], f"Manifest mismatch: {split}")
        for fixture_id, fixture in data["fixtures"].items():
            require(fixture_id not in fixture_splits, f"Fixture crosses split: {fixture_id}")
            fixture_splits[fixture_id] = split
            turn_ids = [t["turn_id"] for t in fixture["history"]]
            require(len(turn_ids) == len(set(turn_ids)), f"Duplicate turn IDs: {fixture_id}")
            for turn in fixture["history"]:
                for source_id in turn["context_source_refs"]:
                    require(source_id in sources, f"Unknown fixture evidence: {source_id}")
                    require(all(sources[source_id][k] == v for k, v in turn["scope"].items()), f"Fixture scope mismatch: {fixture_id}")
        used_fixtures = set()
        for case in data["cases"]:
            require(case["id"] not in all_ids, f"Duplicate case: {case['id']}")
            all_ids.add(case["id"])
            require(case["split"] == split, f"Wrong case split: {case['id']}")
            family = case["scenario_family"]
            require(family_splits.setdefault(family, split) == split, f"Family crosses split: {family}")
            require(manifest["family_split"][family] == split, f"Family manifest mismatch: {family}")
            for fixture_id in case["fixture_sessions"].values():
                require(fixture_id in data["fixtures"], f"Missing fixture: {fixture_id}")
                used_fixtures.add(fixture_id)
                if data["fixtures"][fixture_id]["quality"] == "deliberately_incorrect":
                    require(case["execution_modes"] == ["fixture"], "Fault injection cannot masquerade as a closed-loop run")
            step_ids = [s["step_id"] for s in case["steps"]]
            require(len(step_ids) == len(set(step_ids)), f"Duplicate steps: {case['id']}")
            prior_step = {}
            for step in case["steps"]:
                ex = step["expected"]
                if step["input_kind"] == "clarification_reply":
                    prior = prior_step.get(step["thread"])
                    require(prior and prior["expected"]["behavior"] == "clarify", "Clarification must follow a pending clarification on the same thread")
                prior_step[step["thread"]] = step
                refs = set(ex["required_citation_source_refs"])
                for group in ex["evidence_groups"] + ex.get("conditional_evidence_groups", []):
                    refs.update(group["any_of_source_refs"])
                for source_id in refs:
                    require(source_id in sources, f"Unknown target source: {source_id}")
                    if ex["scope"]:
                        require(all(sources[source_id][k] == v for k, v in ex["scope"].items()), f"Expected evidence in wrong scope: {case['id']}/{step['step_id']}")
                if ex["behavior"] == "clarify":
                    require(ex["scope"] is None and not ex["evidence_groups"], "Clarification cannot require a guessed scope or retrieved evidence")
        require(used_fixtures == set(data["fixtures"]), f"Unused fixtures in {split}")
    require(set(family_splits) == set(manifest["family_split"]), "Orphan family in manifest")


def main():
    schema = read("schema.json")
    datasets = {split: read(f"{split}_eval.json") for split in ["dev", "holdout"]}
    catalog = read("sources.json")
    for split, data in datasets.items():
        check_schema(data, schema, schema, split)
    validate_splits(datasets, catalog, read("split_manifest.json"))
    validate_sources(catalog)
    for split, data in datasets.items():
        print(f"{split}: {len(data['cases'])} conversations, {sum(len(c['steps']) for c in data['cases'])} scored messages")
    print(f"PASS: schema, split/fixture isolation, corpus hashes, {len(catalog['sources'])} source mappings and text anchors")
    print("Visual anchors: author-inspected, not programmatically revalidated. Model evaluation: NOT RUN. Human label review: pending.")


if __name__ == "__main__":
    main()
