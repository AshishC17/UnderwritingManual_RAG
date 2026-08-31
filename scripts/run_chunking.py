"""Run chunking over the source manual and report the size distribution.

Measurement pass: parent-context prepending is deliberately not applied yet — the
per-subsection split counts printed here are what decide that rule.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.chunker import chunk, to_dicts
from src.ingest.parser import parse

DEFAULT_PDF = "data/raw/NovaCred_UW_V0_pdf.pdf"
DEFAULT_OUT = "data/processed/chunks.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    chunks = chunk(parse(args.pdf))
    Path(args.out).write_text(json.dumps(to_dicts(chunks), indent=2))

    counts = [c.token_count for c in chunks]
    print(f"chunks: {len(chunks)}   written to {args.out}")
    print(f"tokens  min={min(counts)}  median={statistics.median(counts):.0f}  "
          f"mean={statistics.mean(counts):.0f}  max={max(counts)}")

    print("\nby chunk_type:")
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
