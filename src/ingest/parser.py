"""PDF -> structured document model: headings, prose, and tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pdfplumber

BODY_SIZE = 10.0
HEADING_LEVELS = {14.0: 1, 12.0: 2, 11.0: 3}
FOOTNOTE_MARKERS = "¹²³⁴⁵⁶⁷⁸⁹"
CAPTION_RE = re.compile(r"^(Table|Exhibit)\s+([A-Z0-9]+)\s*[:—-]?\s*(.*)$", re.I)


@dataclass
class Block:
    kind: str  # heading | prose | table | flowchart
    page: int
    text: str = ""
    level: int | None = None
    rows: list[list[str]] | None = None
    caption: str | None = None
    is_continuation: bool = False


@dataclass
class Document:
    source: str
    blocks: list[Block] = field(default_factory=list)


def _line_items(page):
    """Group words into lines, carrying max font size and bold-ness."""
    words = page.extract_words(extra_attrs=["size", "fontname"])
    rows: dict[float, list] = {}
    for w in words:
        rows.setdefault(round(w["top"], 1), []).append(w)
    out = []
    for top in sorted(rows):
        ws = sorted(rows[top], key=lambda x: x["x0"])
        out.append(
            {
                "top": top,
                "text": " ".join(w["text"] for w in ws).strip(),
                "size": round(max(w["size"] for w in ws), 1),
                "bold": any("Bold" in w["fontname"] for w in ws),
                "x0": min(w["x0"] for w in ws),
                "x1": max(w["x1"] for w in ws),
            }
        )
    return out


def _in_any(line, boxes) -> bool:
    """True if the line's vertical span sits inside a table's bbox."""
    for x0, top, x1, bottom in boxes:
        if top - 2 <= line["top"] <= bottom + 2 and line["x1"] > x0 and line["x0"] < x1:
            return True
    return False


def _clean(rows) -> list[list[str]]:
    cleaned = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    width = max((len(r) for r in cleaned), default=0)
    return [r + [""] * (width - len(r)) for r in cleaned]


def _is_continuation(last: Block | None, rows, idx: int, caption: str | None) -> bool:
    """A continuation repeats the previous table's header, is first on its page, and
    introduces no caption of its own. Sibling tables here share column headers, so the
    header alone cannot distinguish them."""
    if last is None or idx != 0 or caption is not None:
        return False
    return bool(last.rows) and bool(rows) and last.rows[0] == rows[0]


def parse(path: str, captions: dict[int, str] | None = None) -> Document:
    """Parse the PDF. `captions` maps a page number to that page's diagram caption,
    which is inserted as a flowchart block in reading order."""
    doc = Document(source=path.split("/")[-1])
    pending_caption: str | None = None
    last_table: Block | None = None
    captions = captions or {}

    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            found = page.find_tables()
            boxes = [t.bbox for t in found]
            tables = [_clean(t.extract()) for t in found]
            table_tops = [t.bbox[1] for t in found]

            lines = [l for l in _line_items(page) if l["text"]]
            emitted_tables = set()

            for line in lines:
                # Emit any table whose top we have just passed.
                for idx, top in enumerate(table_tops):
                    if idx not in emitted_tables and line["top"] >= top:
                        rows = tables[idx]
                        cont = _is_continuation(last_table, rows, idx, pending_caption)
                        blk = Block(
                            kind="table",
                            page=pno,
                            rows=rows,
                            caption=(last_table.caption if cont else pending_caption),
                            is_continuation=cont,
                        )
                        doc.blocks.append(blk)
                        last_table = blk if not cont else last_table
                        if not cont:
                            pending_caption = None
                        emitted_tables.add(idx)

                if _in_any(line, boxes):
                    continue

                size, text = line["size"], line["text"]
                level = HEADING_LEVELS.get(size) if line["bold"] else None

                if level and len(text) < 90:
                    doc.blocks.append(
                        Block(kind="heading", page=pno, text=text, level=level)
                    )
                    pending_caption = None
                    continue

                if CAPTION_RE.match(text) and len(text) < 90:
                    pending_caption = text
                    continue

                doc.blocks.append(Block(kind="prose", page=pno, text=text))

            # Tables below every line on the page (or on a line-free page).
            for idx, rows in enumerate(tables):
                if idx in emitted_tables:
                    continue
                cont = _is_continuation(last_table, rows, idx, pending_caption)
                blk = Block(
                    kind="table",
                    page=pno,
                    rows=rows,
                    caption=(last_table.caption if cont else pending_caption),
                    is_continuation=cont,
                )
                doc.blocks.append(blk)
                last_table = blk if not cont else last_table
                if not cont:
                    pending_caption = None

            if pno in captions:
                doc.blocks.append(
                    Block(kind="flowchart", page=pno, text=captions[pno])
                )

    return _merge_prose(doc)


def _merge_prose(doc: Document) -> Document:
    """Join consecutive prose lines into paragraph blocks."""
    merged: list[Block] = []
    buf: list[str] = []
    buf_page = 0
    for b in doc.blocks:
        if b.kind == "prose":
            if not buf:
                buf_page = b.page
            buf.append(b.text)
            continue
        if buf:
            merged.append(Block(kind="prose", page=buf_page, text=" ".join(buf)))
            buf = []
        merged.append(b)
    if buf:
        merged.append(Block(kind="prose", page=buf_page, text=" ".join(buf)))
    doc.blocks = merged
    return doc


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = len(rows[0])
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join([" --- "] * width) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def has_footnote_marker(text: str) -> bool:
    return any(m in text for m in FOOTNOTE_MARKERS)
