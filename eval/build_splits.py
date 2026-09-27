"""Build self-contained dev and holdout datasets from the canonical ground truth.

The manifest is the source of split membership. This script validates that the
split is exhaustive, disjoint, and stratified before writing derived files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUND_TRUTH = ROOT / "ground_truth_v1.json"
MANIFEST = ROOT / "split_manifest_v1.json"
OUTPUTS = {
    "dev": ROOT / "dev_eval_v1.json",
    "holdout": ROOT / "holdout_eval_v1.json",
}
QUESTION_OUTPUTS = {
    "dev": ROOT / "dev_questions_v1.json",
    "holdout": ROOT / "holdout_questions_v1.json",
}
READABLE_OUTPUT = ROOT / "qna_pairs_readable_v1.md"


def flatten(split: dict[str, list[str]]) -> list[str]:
    return [case_id for difficulty in split.values() for case_id in difficulty]


def main() -> None:
    dataset = json.loads(GROUND_TRUTH.read_text())
    manifest = json.loads(MANIFEST.read_text())
    by_id = {case["id"]: case for case in dataset["cases"]}

    dev_ids = flatten(manifest["dev"])
    holdout_ids = flatten(manifest["holdout"])
    all_split_ids = dev_ids + holdout_ids

    if set(dev_ids) & set(holdout_ids):
        raise ValueError("Dev and holdout splits overlap")
    if set(all_split_ids) != set(by_id):
        missing = sorted(set(by_id) - set(all_split_ids))
        unknown = sorted(set(all_split_ids) - set(by_id))
        raise ValueError(f"Split is not exhaustive: missing={missing}, unknown={unknown}")
    if len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("A case ID appears more than once in the split manifest")

    dev_counts = Counter(by_id[case_id]["difficulty"] for case_id in dev_ids)
    if dev_counts != {"easy": 3, "medium": 3, "difficult": 3, "extreme": 3}:
        raise ValueError(f"Unexpected dev stratification: {dict(dev_counts)}")

    for split_name, case_ids in (("dev", dev_ids), ("holdout", holdout_ids)):
        selected_cases = [by_id[case_id] for case_id in case_ids]
        output = {
            "schema_version": dataset["schema_version"],
            "split": split_name,
            "source_dataset": GROUND_TRUTH.name,
            "split_manifest": MANIFEST.name,
            "source_document": dataset["source_document"],
            "cases": selected_cases,
        }
        OUTPUTS[split_name].write_text(json.dumps(output, indent=2) + "\n")
        print(f"{split_name}: {len(case_ids)} cases -> {OUTPUTS[split_name].name}")

        question_output = {
            "schema_version": dataset["schema_version"],
            "split": split_name,
            "source_document": dataset["source_document"],
            "cases": [
                {
                    "id": case["id"],
                    "difficulty": case["difficulty"],
                    "question": case["question"],
                }
                for case in selected_cases
            ],
        }
        QUESTION_OUTPUTS[split_name].write_text(
            json.dumps(question_output, indent=2) + "\n"
        )

    split_for_id = {case_id: "dev" for case_id in dev_ids}
    split_for_id.update({case_id: "holdout" for case_id in holdout_ids})
    lines = [
        "# NovaCred Ground-Truth Q&A - Human-Readable View",
        "",
        "This file is for human review. Send only the question to the RAG system; do not send the answer, required claims, forbidden claims, or expected evidence.",
        "",
    ]
    for difficulty in ("easy", "medium", "difficult", "extreme"):
        lines.extend([f"## {difficulty.title()}", ""])
        for case in dataset["cases"]:
            if case["difficulty"] != difficulty:
                continue
            lines.extend(
                [
                    f"### {case['id']} [{split_for_id[case['id']].upper()}]",
                    "",
                    f"**Question:** {case['question']}",
                    "",
                    f"**Ground-truth answer:** {case['ground_truth_answer']}",
                    "",
                    "**A complete answer must say:**",
                    "",
                    *[f"- {claim}" for claim in case["required_claims"]],
                    "",
                    "**The answer must not claim:**",
                    "",
                    *[f"- {claim}" for claim in case["forbidden_claims"]],
                    "",
                    "**Expected evidence groups:**",
                    "",
                ]
            )
            for group in case["evidence_groups"]:
                chunk_ids = ", ".join(
                    chunk_id.split("::")[-1]
                    for chunk_id in group["any_of_chunk_ids"]
                )
                tables = ", ".join(group["tables"]) or "prose/flowchart"
                pages = ", ".join(str(page) for page in group["pages"])
                lines.append(
                    f"- {group['name']}: any of chunks {chunk_ids}; pages {pages}; {tables}"
                )
            lines.extend(["", "---", ""])

    READABLE_OUTPUT.write_text("\n".join(lines))
    print(f"readable: {len(dataset['cases'])} Q&A pairs -> {READABLE_OUTPUT.name}")


if __name__ == "__main__":
    main()
