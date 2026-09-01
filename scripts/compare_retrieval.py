"""Compare dense, sparse (BM25), and hybrid retrieval side by side.

Each retriever fails in a characteristic way, and the failures are opposites:
dense generalises and therefore blurs exact identifiers together; BM25 matches
literally and therefore misses anything phrased differently from the source.
This prints all three rankings for the same query so the difference is visible
rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.embed.embedder import embed_query
from src.embed.sparse import embed_query as embed_query_sparse
from src.store import qdrant_store as qs

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Chosen to expose each retriever's weakness, not to flatter either.
QUERIES = [
    ("exact identifier", "code 128"),
    ("paraphrase", "what happens if the applicant turns down the revised offer"),
    ("table by name", "Table 7 line assignment"),
    ("conceptual", "how are people with no credit history handled"),
]

SPACING_S = 25


def label(hit) -> str:
    p = hit.payload
    name = p.get("table_name") or p["section"]
    return f"[{p['chunk_type'][:7]:7}] p{p['pages'][0]:<3} {name[:40]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    client = qs.connect()

    for i, (kind, q) in enumerate(QUERIES):
        if i:
            time.sleep(SPACING_S)  # free-tier: 3 requests/min
        dv = embed_query(q)
        sv = embed_query_sparse(q)

        print("=" * 78)
        print(f"[{kind}]  {q!r}")

        runs = [
            ("dense ", qs.search(client, dv, limit=args.limit)),
            ("sparse", qs.search_sparse(client, sv, limit=args.limit)),
            ("hybrid", qs.search_hybrid(client, dv, sv, limit=args.limit)),
        ]
        for name, hits in runs:
            print(f"  {name}:")
            for h in hits:
                print(f"    {h.score:6.3f}  {label(h)}")


if __name__ == "__main__":
    main()
