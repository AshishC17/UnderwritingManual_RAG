"""Offline validation only. No application imports, model calls or file writes.

Checks the JSON Schema subset used here, PDF anchors, frozen source mappings,
split isolation, claim/aspect links and controlled fixture evidence coverage.
Does not prove semantic correctness or run the retry pipeline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((HERE / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_schema(value, schema, root=None, path="$"):
    """Fail closed on unknown keywords; intentionally not a general validator."""
    if root is None:
        root = schema
    supported = {"$schema", "$ref", "title", "description", "definitions", "type", "enum",
                 "properties", "additionalProperties", "required", "items",
                 "minItems", "minimum", "maximum"}
    require(not set(schema) - supported, f"Unsupported schema keyword at {path}")
    if "$ref" in schema:
        require(schema["$ref"].startswith("#/"), "Only local schema references supported")
        target = root
        for part in schema["$ref"][2:].split("/"):
            target = target[part]
        return check_schema(value, target, root, path)
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {"null": value is None, "object": isinstance(value, dict),
               "array": isinstance(value, list), "string": isinstance(value, str),
               "boolean": type(value) is bool, "integer": type(value) is int,
               "number": type(value) in (int, float)}
    require(not types or any(matches.get(t, False) for t in types), f"Wrong type: {path}")
    if "enum" in schema:
        require(value in schema["enum"], f"Invalid enum: {path}")
    if type(value) in (int, float):
        require(math.isfinite(value), f"Nonfinite number: {path}")
        require(value >= schema.get("minimum", -math.inf), f"Below minimum: {path}")
        require(value <= schema.get("maximum", math.inf), f"Above maximum: {path}")
    if isinstance(value, dict):
        require(set(schema.get("required", [])) <= set(value), f"Missing keys: {path}")
        for key, item in value.items():
            sub = schema.get("properties", {}).get(key, schema.get("additionalProperties", True))
            require(sub is not False, f"Unknown field: {path}.{key}")
            if isinstance(sub, dict):
                check_schema(item, sub, root, f"{path}.{key}")
    if isinstance(value, list):
        require(len(value) >= schema.get("minItems", 0), f"Too few items: {path}")
        for i, item in enumerate(value):
            check_schema(item, schema.get("items", {}), root, f"{path}[{i}]")


def normalized(text):
    return re.sub(r"\s+", "", text.replace("\u00ad", "").translate(str.maketrans("–—‑", "---")))


def validate_sources(catalog):
    from pypdf import PdfReader
    snap = catalog["corpus_snapshot"]
    files = {snap["manifest_path"]: snap["manifest_sha256"], snap["chunks_path"]: snap["chunks_sha256"]}
    files.update({d["path"]: d["sha256"] for d in snap["documents"]})
    for path, expected in files.items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                f"Snapshot changed: {path}; review/remap sources, never silently relabel.")
    chunks = json.loads((ROOT / snap["chunks_path"]).read_text())
    by_id = {c["chunk_id"]: c for c in chunks}
    require(len(chunks) == len(by_id) == snap["chunk_count"], "Chunk count/ID mismatch")
    readers = {Path(d["path"]).name: PdfReader(ROOT / d["path"]) for d in snap["documents"]}
    text_by_doc = {name: [p.extract_text() or "" for p in reader.pages] for name, reader in readers.items()}
    for ref, source in catalog["sources"].items():
        require(source["chunk_id"] in by_id, f"Unknown source chunk: {ref}")
        chunk = by_id[source["chunk_id"]]
        require(source["pages"] == chunk["pages"], f"Page mismatch: {ref}")
        txt = " ".join(text_by_doc[chunk["source_doc"]][p-1] for p in source["pages"])
        require(normalized(source["anchor"]) in normalized(txt), f"PDF anchor missing: {ref}")
    for audit in catalog["absence_audits"]:
        if audit["literal"] is None:
            continue  # Semantic absence remains an authored, reviewable judgment.
        names = text_by_doc if audit["documents"] == "all_four" else [audit["documents"]]
        pattern = re.escape(audit["literal"])
        if audit["result"] == "absent_whole_word":
            pattern = r"\b" + pattern + r"\b"
        for name in names:
            require(not re.search(pattern, "\n".join(text_by_doc[name]), re.I), f"Absence audit changed: {name}")
    return by_id


def validate_cases(datasets, sources, by_id):
    ids, questions, families = set(), set(), {}
    schema = read("schema.json")
    for dataset in datasets:
        check_schema(dataset, schema)
        split = dataset["split"]
        for case in dataset["cases"]:
            cid = case["id"]
            require(cid not in ids, "Duplicate case ID")
            require(case["question"] not in questions, "Duplicate question across suite")
            ids.add(cid); questions.add(case["question"])
            prior = families.setdefault(case["family"], split)
            require(prior == split, "Question family crosses dev/holdout")
            required_indices = set(range(len(case["required_claims"])))
            aspect_indices = {i for a in case["required_aspects"] for i in a["claim_indices"]}
            require(aspect_indices == required_indices, f"Missing/out-of-range claim link: {cid}")
            require(len({a["id"] for a in case["required_aspects"]}) == len(case["required_aspects"]), "Duplicate aspect")
            require(len({g["name"] for g in case["evidence_groups"]}) == len(case["evidence_groups"]), "Duplicate evidence group")
            if case["answerability"] == "needs_clarification":
                require(case["expected_response"] == "clarify" and not case["evidence_groups"], "Clarification mislabeled")
            elif case["answerability"] in ("unanswerable", "partially_answerable"):
                require(case["expected_response"] == "qualified_answer", "Unknown answer needs explicit limitation")
            else:
                require(case["evidence_groups"] and case["expected_scope"], "Answerable case needs scoped evidence")
            for group in case["evidence_groups"]:
                for ref in group["any_of_source_refs"]:
                    require(ref in sources, f"Unknown source ref: {cid}/{ref}")
                    chunk = by_id[sources[ref]["chunk_id"]]
                    for key, expected in (case["expected_scope"] or {}).items():
                        require(chunk[key] == expected, f"Wrong-scope oracle mapping: {cid}/{ref}")


def context_refs(fixture, catalog):
    refs = list(fixture["context_source_refs"])
    if fixture["pad_to_k"]:
        refs.extend(r for r in catalog["padding_source_refs"] if r not in refs)
    return refs[:catalog["final_k"]]


def covered_groups(case, refs):
    available = set(refs)
    return {g["name"] for g in case["evidence_groups"] if available & set(g["any_of_source_refs"])}


def validate_fixtures(catalog, cases, sources):
    fixtures = {f["id"]: f for f in catalog["fixtures"]}
    require(len(fixtures) == len(catalog["fixtures"]), "Duplicate fixture ID")
    for fixture in fixtures.values():
        require(fixture["case_id"] in cases, "Fixture must reference a dev case")
        case = cases[fixture["case_id"]]
        refs = context_refs(fixture, catalog)
        require(len(refs) == len(set(refs)), "Duplicate context source")
        if fixture["pad_to_k"]:
            require(len(refs) == catalog["final_k"], "Fixture lost its fixed-k control")
        for ref in refs + fixture["outside_context_refs"]:
            require(ref in sources, f"Unknown fixture source: {ref}")
        actual = covered_groups(case, refs)
        require(actual == set(fixture["expected"]["covered_groups"]), f"Fixture oracle drift: {fixture['id']}")
        all_groups = {g["name"] for g in case["evidence_groups"]}
        status = fixture["expected"]["evidence_status"]
        require((actual == all_groups) == status.startswith("sufficient"), "Fixture sufficiency disagrees with groups")
        pairs = set()
        for score in fixture["score_pairs"]:
            require(score["source_ref"] in refs + fixture["outside_context_refs"], "Score not associated with evidence")
            require(type(score["query_index"]) is int and 0 <= score["query_index"] < len(fixture["subqueries"]), "Invalid query association")
            require(type(score["score"]) in (int, float) and math.isfinite(score["score"]), "Invalid score")
            pair = (score["source_ref"], score["query_index"])
            require(pair not in pairs, "Duplicate rerank pair")
            pairs.add(pair)
        require(fixture["attempt"] in (0, 1), "Fixture exceeds one-retry experiment")
        if fixture["retry_of"]:
            previous = fixtures[fixture["retry_of"]]
            require(previous["case_id"] == fixture["case_id"] and previous["attempt"] + 1 == fixture["attempt"], "Invalid retry parent")
        else:
            require(fixture["attempt"] == 0, "Retry needs predecessor")
    for probe in catalog["answer_probes"]:
        require(probe["fixture_id"] in fixtures, "Unknown answer probe fixture")
        case = cases[fixtures[probe["fixture_id"]]["case_id"]]
        expected = probe["expected"]
        for field, claims in [("required_claim_indices_present",case["required_claims"]),("forbidden_claim_indices_present",case["forbidden_claims"])]:
            require(all(type(i) is int and 0 <= i < len(claims) for i in expected[field]), "Invalid probe claim index")
        require(expected["all_required_claims_established"] == (len(set(expected["required_claim_indices_present"])) == len(case["required_claims"])), "Probe completeness mismatch")


def validate_result(record):
    check_schema(record, read("result_record.schema.json"))
    attempts = record["attempts"]
    require([a["attempt"] for a in attempts] == list(range(len(attempts))), "Attempt numbering must start at zero")
    require(len(attempts) <= record["config"]["max_retries"] + 1, "Retry budget exceeded")
    for a in attempts:
        require(a["retry_of"] == (a["attempt"] - 1 if a["attempt"] else None), "Bad retry predecessor")
        require(len(a["context_chunk_ids"]) <= record["config"]["final_k"], "Final-k exceeded")
        n = len(a["queries"])
        for i in a["targeted_query_indices"] + a["assessment"]["unscored_query_indices"]:
            require(0 <= i < n, "Invalid query index")
        for pair in a["rerank_scores"]:
            require(0 <= pair["query_index"] < n, "Bad score query index")
            require(pair["chunk_id"] in a["candidate_chunk_ids"], "Score has no candidate")
    if attempts:
        require(record["final_context_chunk_ids"] == attempts[-1]["context_chunk_ids"], "Final context differs from recorded final attempt")
    if record["status"] in ("error", "not_completed"):
        require(record["metrics"]["pipeline_success"] is not True, "Incomplete run cannot pass")


def self_test(dev, holdout, sources, by_id, fixtures):
    def rejected(callback):
        try:
            callback()
        except (ValueError, KeyError, TypeError):
            return
        raise AssertionError("Invalid fixture unexpectedly accepted")
    bad = copy.deepcopy(dev); bad["cases"][0]["required_aspects"][0]["claim_indices"] = [999]
    rejected(lambda: validate_cases([bad, holdout], sources, by_id))
    bad = copy.deepcopy(holdout); bad["cases"][0]["family"] = dev["cases"][0]["family"]
    rejected(lambda: validate_cases([dev, bad], sources, by_id))
    bad = copy.deepcopy(dev); bad["cases"][0]["evidence_groups"][0]["any_of_source_refs"] = ["missing"]
    rejected(lambda: validate_cases([bad, holdout], sources, by_id))
    cases = {c["id"]: c for c in dev["cases"]}
    bad = copy.deepcopy(fixtures); bad["fixtures"][3]["expected"]["covered_groups"].append("override")
    rejected(lambda: validate_fixtures(bad, cases, sources))
    bad = copy.deepcopy(fixtures); bad["fixtures"][6]["attempt"] = 2
    rejected(lambda: validate_fixtures(bad, cases, sources))
    rejected(lambda: check_schema(True, {"type":"integer"}))
    rejected(lambda: check_schema(float("nan"), {"type":"number"}))
    require(covered_groups(cases["R-D02"], context_refs(fixtures["fixtures"][3],fixtures)) == {"rule"}, "Rank-11 evidence leaked into final context")
    before = covered_groups(cases["R-D02"], context_refs(fixtures["fixtures"][2],fixtures))
    after = covered_groups(cases["R-D02"], context_refs(fixtures["fixtures"][9],fixtures))
    require(before - after == {"rule"} and after - before == {"override"}, "Evidence-loss regression not detected")
    print("9 offline negative/control checks passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--result", type=Path, help="Validate one recorded result; does not execute a model")
    args = ap.parse_args()
    dev, holdout, sources, fixtures = (read(n) for n in ("dev_eval.json","holdout_eval.json","sources.json","gate_fixtures.json"))
    by_id = validate_sources(sources)
    validate_cases([dev,holdout], sources["sources"], by_id)
    validate_fixtures(fixtures, {c["id"]:c for c in dev["cases"]}, sources["sources"])
    print(f"Validated {len(dev['cases'])} dev / {len(holdout['cases'])} holdout questions, {len(fixtures['fixtures'])} gate fixtures, {len(fixtures['answer_probes'])} judge probes, {len(sources['sources'])} PDF source mappings.")
    if args.self_test:
        self_test(dev, holdout, sources["sources"], by_id, fixtures)
    if args.result:
        validate_result(json.loads(args.result.read_text()))
        print("Result structure and attempt bounds validated; semantic scores were not audited.")


if __name__ == "__main__":
    main()
