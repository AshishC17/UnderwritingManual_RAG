"""Chunker-independent evidence matching: is an excerpt's text present in the chunks?"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

MIN_RUN = 4
_TOKEN = re.compile(r"[a-z0-9]+|[<>≤≥=$%+]")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def covered(excerpt_tokens: list[str], chunk_tokens: list[str]) -> set[int]:
    """Excerpt token positions found in the chunk inside runs of at least MIN_RUN tokens."""
    floor = min(MIN_RUN, len(excerpt_tokens))
    matcher = difflib.SequenceMatcher(None, excerpt_tokens, chunk_tokens, autojunk=False)
    hit: set[int] = set()
    for block in matcher.get_matching_blocks():
        if block.size >= floor:
            hit.update(range(block.a, block.a + block.size))
    return hit


@dataclass
class Assessment:
    status: str
    coverage: float
    intact_in: list[str] = field(default_factory=list)
    parts: dict[str, float] = field(default_factory=dict)


def assess(text: str, chunks: list[dict]) -> Assessment:
    """intact: one chunk holds it all; split: only the chunks together do; partial; missing."""
    excerpt = tokens(text)
    if not excerpt:
        raise ValueError("excerpt has no matchable tokens")
    union: set[int] = set()
    intact_in: list[str] = []
    parts: dict[str, float] = {}
    for chunk in chunks:
        hit = covered(excerpt, tokens(chunk["text"]))
        if not hit:
            continue
        union |= hit
        parts[chunk["chunk_id"]] = len(hit) / len(excerpt)
        if len(hit) == len(excerpt):
            intact_in.append(chunk["chunk_id"])
    coverage = len(union) / len(excerpt)
    if intact_in:
        status = "intact"
    elif coverage == 1.0:
        status = "split"
    elif coverage > 0:
        status = "partial"
    else:
        status = "missing"
    return Assessment(status, coverage, intact_in, parts)


def identified_in(text: str, context: str | None, chunks: list[dict]) -> list[str]:
    """Chunks holding the whole excerpt and, when given, the context that says which table or chart it is."""
    with_text = assess(text, chunks).intact_in
    if not context:
        return with_text
    with_context = set(assess(context, chunks).intact_in)
    return [chunk_id for chunk_id in with_text if chunk_id in with_context]


def assess_by_document(text: str, chunks: list[dict]) -> dict[str, Assessment]:
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["source_doc"], []).append(chunk)
    return {doc: assess(text, group) for doc, group in by_doc.items()}


def anchor_tokens(text: str) -> list[str]:
    """V2 preserves decimal/range signs, unlike the original exploratory matcher."""
    return re.findall(r"\d+(?:[.,]\d+)*|[a-z]+|[<>≤≥=$%+\-]",
                      unicodedata.normalize("NFKC", text).lower())


def contains_anchor(needle: str, haystack: str) -> bool:
    a, b = anchor_tokens(needle), anchor_tokens(haystack)
    return bool(a) and any(b[i:i+len(a)] == a for i in range(len(b)-len(a)+1))


def _table_row_intact(item: dict, text: str) -> bool:
    """Match ordered cells under the nearest header and correct table caption."""
    def cells(line):
        if not line.strip().startswith("|") or not line.strip().endswith("|"):
            return None
        return [anchor_tokens(cell.strip()) for cell in line.strip().strip("|").split("|")]
    wanted = cells(item["text"])
    if wanted is None:
        return False
    expected_headers = [cells(h) for h in item.get("required_headers", [])]
    lines = text.splitlines()
    header, caption = None, None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(?:#+\s*)?(?:Table|Exhibit)\s+[a-z0-9]+\s*[:.]", line, re.I):
            caption, header = line, None
        if i and re.fullmatch(r"\s*\|[\s:|\-]+\|\s*", line):
            header = cells(lines[i-1])
        if cells(line) == wanted:
            if expected_headers and (header not in expected_headers or len(header) != len(wanted)):
                continue
            if item.get("context") and (not caption or not contains_anchor(item["context"], caption)):
                continue
            return True
    return False


def evidence_result(item: dict, chunks: list[dict]) -> dict:
    """Source-anchored *lexical* coverage, not a semantic/answer-quality judge.

    Does not read authoring_chunk_id. A table row must be intact, under its
    declared header and caption in the same chunk. Prose may span chunks.
    Supporting IDs are a greedy sufficient witness, NOT an exact minimum.
    """
    own = [c for c in chunks if c.get("source_doc") == item["document"]]
    for key, value in item.get("source_scope", {}).items():
        own = [c for c in own if c.get(key) == value]
    required = [s for s in [item.get("context"), *item.get("required_headers", [])] if s]
    eligible = [c for c in own if all(contains_anchor(s, c["text"]) for s in required)]
    a = anchor_tokens(item["text"])
    if not a:
        raise ValueError("empty excerpt")
    is_table = item.get("kind") == "table_row"
    intact = [c["chunk_id"] for c in eligible if (
        _table_row_intact(item, c["text"]) if is_table else contains_anchor(item["text"], c["text"]))]
    if intact:
        return {"status": "intact", "coverage": 1., "resolved": True,
                "chunk_ids": [sorted(intact)[0]], "all_matching_chunk_ids": sorted(intact)}
    # Never assemble a table row from different rows/cells and call it valid.
    parts = {c["chunk_id"]: covered(a, anchor_tokens(c["text"])) for c in eligible}
    remaining, selected = set(range(len(a))), []
    while remaining and parts:
        cid = max(sorted(parts), key=lambda k: len(parts[k] & remaining))
        if not (parts[cid] & remaining):
            break
        remaining -= parts.pop(cid)
        selected.append(cid)
    ratio = 1 - len(remaining) / len(a)
    wrong_context = any(contains_anchor(item["text"], c["text"]) for c in own)
    status = ("wrong_context" if wrong_context else "split" if ratio == 1 and not is_table
              else "partial" if ratio else "missing")
    return {"status": status, "coverage": ratio, "resolved": status in {"intact", "split"},
            "chunk_ids": selected, "all_matching_chunk_ids": sorted(c["chunk_id"] for c in eligible
                if covered(a, anchor_tokens(c["text"])))}


def map_evidence(document: dict, chunks: list[dict]) -> dict:
    mapped = {eid: evidence_result(item, chunks) for eid, item in document["excerpts"].items()}
    groups = []
    for group in document["groups"]:
        ok = any(alt["excerpts"] and all(mapped[e]["resolved"] for e in alt["excerpts"])
                 for alt in group["any_of"])
        groups.append({"case": group["case"], "group": group["group"], "covered": ok})
    return {"excerpts": mapped, "groups": groups,
            "group_coverage": sum(g["covered"] for g in groups)/len(groups) if groups else None,
            "all_groups_covered": bool(groups) and all(g["covered"] for g in groups),
            "qualification": "lexical and declared-structure checks; not semantic sufficiency"}
