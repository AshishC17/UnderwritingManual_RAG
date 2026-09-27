"""Run structure-aware chunking over every document in the corpus manifest.

Measurement pass: parent-context prepending is deliberately not applied yet — the
per-subsection split counts printed here are what decide that rule.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.chunker import chunk, to_dicts
from src.ingest.flowchart import caption, extract_images
from src.ingest.manifest import DocumentSpec, assign_chunk_revisions, load_manifest
from src.ingest.parser import parse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "config/corpus_manifest.json"
DEFAULT_OUT = "data/processed/chunks_v2.json"
IMAGE_DIR = "data/interim/images"
CAPTION_DIR = "data/interim/captions"


def _authored_captions(spec: DocumentSpec) -> dict[int, str]:
    path = spec.resolved_caption_file(PROJECT_ROOT)
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    from src.guardrails.checks import evidence_check
    evidence_check([{"text": text} for text in raw.values()])
    return {int(page): text.strip() for page, text in raw.items()}


def load_captions(
    pdf: str,
    spec: DocumentSpec,
    *,
    allow_api: bool,
    allow_missing: bool,
) -> dict[int, str]:
    """Load deterministic sidecars first, then the hash cache/API if necessary."""
    authored = _authored_captions(spec)
    captions: dict[int, str] = {}
    missing: list[str] = []
    image_dir = Path(IMAGE_DIR) / Path(pdf).stem
    refs = extract_images(pdf, str(image_dir))
    image_pages = {ref.page for ref in refs}

    unknown_pages = set(authored) - image_pages
    if unknown_pages:
        raise ValueError(
            f"{spec.caption_file} has captions for pages without images: "
            f"{sorted(unknown_pages)}"
        )

    for ref in refs:
        text = authored.get(ref.page) or caption(
            ref, CAPTION_DIR, allow_api=allow_api
        )
        if text:
            captions[ref.page] = text
        else:
            missing.append(f"page {ref.page} (sha {ref.sha})")

    if missing and not allow_missing:
        raise RuntimeError(
            f"{spec.source_doc} has uncaptained diagrams: {', '.join(missing)}. "
            "Add a caption sidecar, configure the vision API, or explicitly pass "
            "--allow-missing-captions."
        )
    for item in missing:
        print(f"  WARNING: {spec.source_doc} {item} has no caption")
    return captions


def _report(chunks, label: str) -> None:
    counts = [c.token_count for c in chunks]
    print(
        f"{label}: {len(chunks)} chunks; tokens min={min(counts)} "
        f"median={statistics.median(counts):.0f} "
        f"mean={statistics.mean(counts):.0f} max={max(counts)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--pdf",
        help="process only this manifest PDF (useful for compatibility checks)",
    )
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--no-caption-api",
        action="store_true",
        help="use only authored sidecars and the existing image-hash cache",
    )
    ap.add_argument(
        "--allow-missing-captions",
        action="store_true",
        help="continue even when an embedded diagram cannot be represented as text",
    )
    args = ap.parse_args()

    specs = load_manifest(args.manifest, root=PROJECT_ROOT)
    if args.pdf:
        selected = Path(args.pdf).resolve()
        specs = [
            spec
            for spec in specs
            if spec.resolved_path(PROJECT_ROOT).resolve() == selected
            or spec.source_doc == Path(args.pdf).name
        ]
        if not specs:
            raise SystemExit(f"{args.pdf} is not listed in {args.manifest}")

    chunks = []
    total_captions = 0
    for spec in specs:
        pdf = str(spec.resolved_path(PROJECT_ROOT))
        captions = load_captions(
            pdf,
            spec,
            allow_api=not args.no_caption_api,
            allow_missing=args.allow_missing_captions,
        )
        total_captions += len(captions)
        document_chunks = chunk(
            parse(pdf, captions=captions), metadata=spec.chunk_metadata()
        )
        chunks.extend(document_chunks)
        _report(document_chunks, spec.source_doc)

    assign_chunk_revisions(chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_dicts(chunks), indent=2))

    counts = [c.token_count for c in chunks]
    print(f"\ndiagram captions loaded: {total_captions}")
    print(f"chunks: {len(chunks)}   written to {out}")
    print(f"tokens  min={min(counts)}  median={statistics.median(counts):.0f}  "
          f"mean={statistics.mean(counts):.0f}  max={max(counts)}")

    print("\nby chunk_type across corpus:")
    for k, v in Counter(c.chunk_type for c in chunks).most_common():
        sub = [c.token_count for c in chunks if c.chunk_type == k]
        print(f"  {k:<10} n={v:<4} median={statistics.median(sub):.0f}  max={max(sub)}")

    print("\noversized (> 512):")
    over = [c for c in chunks if c.token_count > 512]
    for c in over:
        print(f"  {c.chunk_id} {c.token_count:>5}  {c.chunk_type:<9} {c.table_name or c.subsection or c.section}")
    if not over:
        print("  none")

    # The decision this run exists to inform.
    per_sub: dict[tuple[str, str | None], list] = defaultdict(list)
    for c in chunks:
        per_sub[(c.section, c.subsection)].append(c)

    single = [k for k, v in per_sub.items() if len(v) == 1]
    multi = [k for k, v in per_sub.items() if len(v) > 1]
    print(f"\nsubsections producing 1 chunk: {len(single)}   2+ chunks: {len(multi)}")
    print("\nchunk counts per subsection:")
    for (sec, sub), v in sorted(per_sub.items(), key=lambda kv: -len(kv[1])):
        toks = sum(c.token_count for c in v)
        label = f"{sec} > {sub}" if sub else sec
        print(f"  {len(v):>3} chunks {toks:>6} tok  {label[:66]}")


if __name__ == "__main__":
    main()
