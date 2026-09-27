"""Validate comparison eval labels against the current indexed chunk snapshot.

No model calls and no application execution. This catches stale chunk IDs,
wrong lender/version labels, duplicate questions, malformed follow-up fixtures,
and accidental edits to the frozen holdout contract.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def key(scope: dict) -> str:
    return "::".join(scope[field] for field in (
        "lender_id", "document_id", "document_version"
    ))


def validate() -> dict:
    dev, holdout = read("dev_eval.json"), read("holdout_eval.json")
    chunks = json.loads((ROOT / "data/processed/chunks_v2.json").read_text())
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    require(len(chunks) == len(by_id), "chunk IDs are not unique")
    require(dev["split"] == "dev" and not dev["frozen"], "dev split contract changed")
    require(holdout["split"] == "holdout" and holdout["frozen"], "holdout must remain frozen")

    ids: set[str] = set()
    scenarios: set[str] = set()
    families: dict[str, str] = {}
    counts = {"dev": 0, "holdout": 0}
    for dataset in (dev, holdout):
        require(dataset["schema_version"] == "1.0", "unknown schema version")
        require(dataset["suite"] == "comparison_eval_v1", "wrong suite")
        split = dataset["split"]
        for case in dataset["cases"]:
            cid = case["id"]
            require(cid not in ids, f"duplicate case id: {cid}")
            scenario = json.dumps(
                {"question": case["question"], "setup": case["setup"]},
                sort_keys=True,
            )
            require(scenario not in scenarios, f"duplicate conversation scenario: {cid}")
            require(families.setdefault(case["family"], split) == split,
                    f"family crosses dev/holdout: {case['family']}")
            ids.add(cid); scenarios.add(scenario); counts[split] += 1

            expected = case["expected"]
            scopes = expected["scopes"]
            scope_keys = {key(scope) for scope in scopes}
            require(len(scope_keys) == len(scopes), f"duplicate expected scope: {cid}")
            if expected["outcome"] == "clarification":
                require(expected["behavior"] == "clarify", f"clarification route mismatch: {cid}")
                require(not scopes and not expected["evidence_groups"],
                        f"clarification must not carry retrieval labels: {cid}")
            else:
                require(expected["behavior"] == "compare_policy", f"answer route mismatch: {cid}")
                require(len(scopes) == 2 and len(expected["required_claims"]) > 0,
                        f"answer case needs two scopes and claims: {cid}")

            group_names: set[str] = set()
            for group in expected["evidence_groups"]:
                require(group["name"] not in group_names, f"duplicate group: {cid}")
                group_names.add(group["name"])
                require(group["scope_key"] in scope_keys, f"group has unknown scope: {cid}")
                for chunk_id in group["any_of_chunk_ids"]:
                    require(chunk_id in by_id, f"unknown chunk: {cid}/{chunk_id}")
                    chunk = by_id[chunk_id]
                    actual = "::".join(str(chunk[field]) for field in (
                        "lender_id", "document_id", "document_version"
                    ))
                    require(actual == group["scope_key"],
                            f"wrong-scope evidence label: {cid}/{chunk_id}")

            setup = case["setup"]
            if setup["prior_scope"] is None:
                require(not setup["history"], f"history without prior scope: {cid}")
            else:
                require(setup["history"], f"prior scope without history: {cid}")
                require(key(setup["prior_scope"]) == key(setup["history"][-1]["scope"]),
                        f"prior scope does not match last fixture turn: {cid}")
                for turn in setup["history"]:
                    for chunk_id in turn["chunk_ids"]:
                        require(chunk_id in by_id, f"unknown fixture chunk: {cid}/{chunk_id}")

    return {"status": "ok", "cases": counts, "chunks_checked": len(by_id)}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
