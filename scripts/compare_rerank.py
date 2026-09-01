"""Show what reranking changes: hybrid's ordering before vs after.

Prints the shortlist hybrid retrieval produced, then the order the cross-encoder
puts it in, with movement marked. The interesting cases are chunks that hybrid
buried and the reranker promoted — those are the ones a top-5 cutoff would
otherwise have thrown away.
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
from src.rerank.reranker import rerank
from src.store import qdrant_store as qs

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

QUERIES = [
    "what does code 128 mean",
    "what happens if the applicant turns down the revised offer",
]

SPACING_S = 30


def label(payload: dict) -> str:
    name = payload.get("table_name") or payload["section"]
    return f"[{payload['chunk_type'][:7]:7}] p{payload['pages'][0]:<3} {name[:38]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    client = qs.connect()

    for i, q in enumerate(QUERIES):
        if i:
            time.sleep(SPACING_S)

        hits = qs.search_hybrid(
            client,
            embed_query(q),
            embed_query_sparse(q),
            limit=args.candidates,
            prefetch_limit=args.candidates * 2,
        )
        before = [h.payload for h in hits]
        order = {id(p): n for n, p in enumerate(before)}

        time.sleep(SPACING_S)
        after = rerank(q, before)

        print("=" * 78)
        print(f"{q!r}   ({len(before)} candidates -> reranked)")
        print(f"\n  hybrid order (RRF rank):")
        for n, (h, p) in enumerate(zip(hits[: args.show], before[: args.show])):
            print(f"    {n + 1:>2}. {h.score:.3f}  {label(p)}")

        print(f"\n  after reranking (absolute relevance):")
        for n, (p, score) in enumerate(after[: args.show]):
            was = order[id(p)]
            move = "  " if was == n else ("up" if was > n else "dn")
            delta = f"{move} {abs(was - n)}" if was != n else "  ="
            print(f"    {n + 1:>2}. {score:.3f}  {label(p)}   ({delta}, was #{was + 1})")


if __name__ == "__main__":
    main()
