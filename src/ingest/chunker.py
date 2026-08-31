"""Document model -> retrieval chunks.

Structure-aware: prose is split to a token budget, tables stay whole where they fit
and repeat their header row when they do not.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import tiktoken

from .parser import Block, Document, has_footnote_marker, table_to_markdown

MAX_TOKENS = 512
OVERLAP_RATIO = 0.15
CODE_RE = re.compile(r"\b(1[0-4][0-9])\b")
CRITERION_RE = re.compile(r"\b(FC-\d{2})\b")

_enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc.encode(text))


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


def chunk(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    pos = 0

    for grp in _group(doc):
        blocks = _stitch(grp.blocks)
        ctype_base = _classify(grp.section)

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
                    )
                )
                pos += 1

    return chunks


def to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
