"""Flowchart/diagram handling: extract embedded images, caption them to text.

Captioning is cached on disk, keyed by image content hash. Vision calls are the
most expensive step in ingestion and the source images rarely change, so a cached
caption is reused until the image itself does.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

MODEL = "claude-opus-5"

CAPTION_PROMPT = """This is a process flowchart from an underwriting policy manual.
Transcribe it completely as text for a retrieval system. Output exactly:

Line 1: the flowchart's title.
Then a paragraph describing the process narratively, naming every node.
Then a line `Decision paths:` followed by one line per edge in the form
`- Source --label--> Target` (omit `--label--` where an edge is unlabelled).

Preserve node labels verbatim, including terminal states. Do not add commentary."""


@dataclass
class ImageRef:
    path: Path
    page: int
    sha: str


def extract_images(pdf_path: str, out_dir: str) -> list[ImageRef]:
    """Pull embedded raster images out of the PDF, one file per image."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf_path)
    refs: list[ImageRef] = []

    for pno, page in enumerate(reader.pages, start=1):
        for idx, img in enumerate(page.images):
            data = img.data
            sha = hashlib.sha256(data).hexdigest()[:16]
            dest = out / f"p{pno:02d}_img{idx}.png"
            dest.write_bytes(data)
            refs.append(ImageRef(path=dest, page=pno, sha=sha))
    return refs


def _cache_path(cache_dir: str, sha: str) -> Path:
    return Path(cache_dir) / f"{sha}.md"


def caption(ref: ImageRef, cache_dir: str, allow_api: bool = True) -> str | None:
    """Return the caption for an image, calling the API only on a cache miss.

    Returns None when the caption is absent and no API call can be made, so the
    caller can report the gap rather than silently indexing an empty diagram.
    """
    cached = _cache_path(cache_dir, ref.sha)
    if cached.exists():
        return cached.read_text().strip()

    if not allow_api:
        return None

    text = _call_vision(ref)
    if text:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
    return text


def _call_vision(ref: ImageRef) -> str | None:
    """Caption one image with Claude. Requires credentials in the environment."""
    try:
        import anthropic
    except ImportError:
        return None

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None

    client = anthropic.Anthropic()
    b64 = base64.standard_b64encode(ref.path.read_bytes()).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
