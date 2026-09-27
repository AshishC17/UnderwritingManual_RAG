"""Document model -> retrieval chunks.

Structure-aware: prose is split to a token budget, tables stay whole where they fit
and repeat their header row when they do not.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken

from .parser import Block, Document, has_footnote_marker, table_to_markdown

MAX_TOKENS = 512
OVERLAP_RATIO = 0.15
CODE_RE = re.compile(r"\b(1[0-4][0-9])\b")
CRITERION_RE = re.compile(r"\b(FC-\d{2})\b")
VERSION_WORD_RE = re.compile(r"\bversion\s+\d+\b|\bv\d+\b", re.I)
TABLE_ID_RE = re.compile(r"^(table|exhibit)\s+([a-z0-9]+)\b", re.I)


@lru_cache(maxsize=1)
def _encoder():
    """Load the tokenizer only when chunk sizing is actually requested."""
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@dataclass
class Chunk:
    chunk_id: str
    text: str
    chunk_type: str
    section: str
    subsection: str | None
    pages: list[int]
    table_name: str | None
    rule_codes_referenced: list[str]
    has_footnote: bool
    source_doc: str
    position_in_doc: int
    token_count: int
    parent_context: str | None = None
    lender_id: str | None = None
    lender_name: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    document_family: str | None = None
    document_id: str | None = None
    document_version: str | None = None
    version_order: int | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    version_status: str | None = None
    supersedes_version: str | None = None
    authority_level: str | None = None
    section_key: str | None = None
    subsection_key: str | None = None
    lineage_id: str | None = None
    chunk_revision: int = 1
    content_hash: str | None = None
    chunking_version: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_version: str | None = None


@dataclass
class _Section:
    section: str
    subsection: str | None
    blocks: list[Block] = field(default_factory=list)


def _codes(text: str) -> list[str]:
    return sorted(set(CODE_RE.findall(text)) | set(CRITERION_RE.findall(text)))


def _classify(section: str) -> str:
    upper = section.upper()
    if "EXHIBIT B" in upper:
        return "flowchart"
    if upper.startswith("EXHIBIT") or upper.startswith("APPENDIX"):
        return "appendix"
    return "primary"


def _stable_key(text: str | None) -> str | None:
    """Normalize a structural label into a version-independent metadata key."""
    if text is None:
        return None
    text = VERSION_WORD_RE.sub("", text)
    text = re.sub(r"\bpre[\s-]?qual\b", "prequalification", text, flags=re.I)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    key = re.sub(r"[^a-z0-9]+", ".", text.lower()).strip(".")
    return key or "untitled"


def _lineage_anchor(kind: str, table_name: str | None, text: str) -> str:
    if kind == "table" and table_name:
        match = TABLE_ID_RE.match(table_name)
        if match:
            return f"{match.group(1).lower()}.{match.group(2).lower()}"
        return _stable_key(table_name) or "table"
    if kind == "flowchart":
        return _stable_key(text.splitlines()[0] if text else None) or "flowchart"
    return "prose"


def _split_prose(text: str) -> list[str]:
    """Sentence-aware split to MAX_TOKENS with OVERLAP_RATIO carry-over."""
    if n_tokens(text) <= MAX_TOKENS:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    overlap_budget = int(MAX_TOKENS * OVERLAP_RATIO)
    out: list[str] = []
    cur: list[str] = []

    for sent in sentences:
        trial = " ".join(cur + [sent])
        if cur and n_tokens(trial) > MAX_TOKENS:
            out.append(" ".join(cur))
            carry: list[str] = []
            for prev in reversed(cur):
                if n_tokens(" ".join([prev] + carry)) > overlap_budget:
                    break
                carry.insert(0, prev)
            cur = carry + [sent]
        else:
            cur.append(sent)
    if cur:
        out.append(" ".join(cur))
    return out


def _split_table(rows: list[list[str]], caption: str | None) -> list[str]:
    """Whole table if it fits; otherwise row-groups, each repeating the header.

    No overlap between table chunks: rows are independent records, so repeating them
    would duplicate data without helping retrieval.
    """
    header, body = rows[0], rows[1:]
    prefix = f"{caption}\n" if caption else ""
    whole = prefix + table_to_markdown(rows)
    if n_tokens(whole) <= MAX_TOKENS:
        return [whole]

    out, cur = [], []
    for row in body:
        trial = prefix + table_to_markdown([header] + cur + [row])
        if cur and n_tokens(trial) > MAX_TOKENS:
            out.append(prefix + table_to_markdown([header] + cur))
            cur = [row]
        else:
            cur.append(row)
    if cur:
        out.append(prefix + table_to_markdown([header] + cur))
    return out


def _split_flowchart(text: str) -> list[str]:
    """Split a diagram caption between its narrative and its edge list.

    The title is repeated on the edge-list chunk: an edge like
    `Rule Result? --Pass--> Assign Line` is ambiguous without knowing which of the
    two near-identical process diagrams it belongs to.
    """
    if n_tokens(text) <= MAX_TOKENS:
        return [text]

    marker = "Decision paths:"
    if marker not in text:
        return _split_prose(text)

    narrative, edges = text.split(marker, 1)
    title = narrative.strip().split("\n", 1)[0]
    head = narrative.strip()
    tail = f"{title}\n{marker}{edges.rstrip()}"

    out = []
    for part in (head, tail):
        out.extend(_split_prose(part) if n_tokens(part) > MAX_TOKENS else [part])
    return out


def _stitch(blocks: list[Block]) -> list[Block]:
    """Merge cross-page table continuations back into one logical table."""
    out: list[Block] = []
    for b in blocks:
        if b.kind == "table" and b.is_continuation and out and out[-1].kind == "table":
            out[-1].rows = out[-1].rows + b.rows[1:]
            continue
        out.append(b)
    return out


def _group(doc: Document) -> list[_Section]:
    """Group blocks under their nearest heading, tracking the parent section."""
    groups: list[_Section] = []
    section = doc.source
    current: _Section | None = None

    for b in doc.blocks:
        if b.kind == "heading":
            if b.level in (1, 2):
                section = b.text
                current = _Section(section=section, subsection=None)
            else:
                current = _Section(section=section, subsection=b.text)
            groups.append(current)
            continue
        if current is None:
            current = _Section(section=section, subsection=None)
            groups.append(current)
        current.blocks.append(b)
    return groups


def chunk(doc: Document, metadata: dict[str, Any] | None = None) -> list[Chunk]:
    """Create chunks and attach document/version metadata when supplied.

    ``chunk_id`` deliberately keeps the original source-file-plus-position format.
    ``lineage_id`` is structural and version independent, so it can match the same
    logical chunk in V0 and V1 even when their positional IDs differ.
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    lineage_counts: dict[tuple[str, str | None, str, str], int] = defaultdict(int)
    pos = 0
    document_id = metadata.get("document_id") or Path(doc.source).stem

    for grp in _group(doc):
        blocks = _stitch(grp.blocks)
        ctype_base = _classify(grp.section)
        section_key = _stable_key(grp.section) or "untitled"
        subsection_key = _stable_key(grp.subsection)

        for b in blocks:
            if b.kind == "table":
                texts = _split_table(b.rows, b.caption)
                kind = "table"
                table_name = b.caption
            elif b.kind == "flowchart":
                texts = _split_flowchart(b.text)
                kind = "flowchart"
                table_name = None
            else:
                texts = _split_prose(b.text)
                kind = ctype_base
                table_name = None

            for t in texts:
                anchor = _lineage_anchor(kind, table_name, t)
                # Tables and diagrams have their own stable named identity; their
                # parent heading may move between versions without changing them.
                lineage_section = "_" if kind in {"table", "flowchart"} else section_key
                lineage_subsection = None if kind in {"table", "flowchart"} else subsection_key
                lineage_key = (lineage_section, lineage_subsection, kind, anchor)
                lineage_counts[lineage_key] += 1
                lineage_id = (
                    f"{document_id}::{lineage_section}::{lineage_subsection or '_'}::"
                    f"{kind}::{anchor}::{lineage_counts[lineage_key]:03d}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.source}::{pos:04d}",
                        text=t,
                        chunk_type=kind,
                        section=grp.section,
                        subsection=grp.subsection,
                        pages=[b.page],
                        table_name=table_name,
                        rule_codes_referenced=_codes(t),
                        has_footnote=has_footnote_marker(t),
                        source_doc=doc.source,
                        position_in_doc=pos,
                        token_count=n_tokens(t),
                        lender_id=metadata.get("lender_id"),
                        lender_name=metadata.get("lender_name"),
                        product_id=metadata.get("product_id"),
                        product_name=metadata.get("product_name"),
                        document_family=metadata.get("document_family"),
                        document_id=document_id,
                        document_version=metadata.get("document_version"),
                        version_order=metadata.get("version_order"),
                        effective_from=metadata.get("effective_from"),
                        effective_to=metadata.get("effective_to"),
                        version_status=metadata.get("version_status"),
                        supersedes_version=metadata.get("supersedes_version"),
                        authority_level=metadata.get("authority_level"),
                        section_key=section_key,
                        subsection_key=subsection_key,
                        lineage_id=lineage_id,
                        content_hash=hashlib.sha256(t.encode("utf-8")).hexdigest(),
                        chunking_version=metadata.get("chunking_version"),
                    )
                )
                pos += 1

    return chunks


def to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
