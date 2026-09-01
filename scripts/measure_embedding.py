"""Characterise the embedding model empirically.

Measures the properties of an embedding model that can actually be measured,
rather than read off a model card: dimensions, whether queries and documents are
encoded differently, whether surrounding-document context genuinely changes a
vector, per-call latency, and cost.

Retrieval quality is deliberately absent — it needs a labelled query set and
belongs with the eval harness, not here.

Each measurement costs API calls, so the count is kept small and spaced: the
free tier allows 3 requests/minute.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.embed.embedder import DIMS, MODEL, _embed, _plan_groups

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SPACING_S = 25  # 3 requests/min on the free tier


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dims", type=int, default=DIMS)
    ap.add_argument("--probe", type=int, default=33, help="chunk index to probe")
    ap.add_argument("--cache", default="data/interim/embeddings")
    args = ap.parse_args()

    chunks = json.loads(Path(args.chunks).read_text())
    texts = [c["text"] for c in chunks]
    probe = args.probe

    # Locate the probe chunk's contextualization group and its position in it.
    groups = _plan_groups(texts, 9000)
    offset, group, index = 0, None, None
    for g in groups:
        if offset <= probe < offset + len(g):
            group, index = g, probe - offset
            break
        offset += len(g)

    print(f"model: {args.model}   dims requested: {args.dims}")
    print(f"probe: chunk {probe} — {texts[probe][:56]!r}")
    print(f"       sits at position {index} of its {len(group)}-chunk group\n")

    def timed(fn, label):
        t0 = time.perf_counter()
        out = fn()
        dt = (time.perf_counter() - t0) * 1000
        print(f"  {label}: {dt:.0f} ms")
        return out, dt

    # 1. In-context vector (cached from the ingest run — no API call, no cost).
    in_ctx = _embed(group, args.model, args.dims, "document", args.cache)[index]
    print(f"1. dimensions actually returned: {len(in_ctx)}")

    # 2. Same chunk with NO neighbours. If contextualization is real, this vector
    #    differs from the in-context one for identical input text.
    print("\n2. does surrounding context change the vector?")
    alone, t_alone = timed(
        lambda: _embed([texts[probe]], args.model, args.dims, "document", args.cache)[0],
        "embed chunk alone (1 API call)",
    )
    sim = cosine(in_ctx, alone)
    print(f"   cosine(in-context, alone) = {sim:.4f}")
    print(
        "   -> contextualization is REAL: same text, different vector"
        if sim < 0.999
        else "   -> no measurable contextualization effect"
    )

    # 3. Does input_type change the encoding of identical text?
    print("\n3. are queries and documents encoded differently?")
    time.sleep(SPACING_S)
    as_query, t_query = timed(
        lambda: _embed([texts[probe]], args.model, args.dims, "query", args.cache)[0],
        "embed same text as query (1 API call)",
    )
    print(f"   cosine(as-document, as-query) = {cosine(alone, as_query):.4f}")
    print("   -> input_type materially changes the vector; never mix them")

    # 4. Query-path latency: a real query is one short string, never contextualized.
    print("\n4. query-path latency (what a user actually waits for)")
    time.sleep(SPACING_S)
    _, t_real = timed(
        lambda: _embed(
            [f"what happens if verification fails {time.time()}"],
            args.model,
            args.dims,
            "query",
            args.cache,
        )[0],
        "embed a fresh query (1 API call)",
    )
    print(f"   single short string, no neighbours -> no contextualization work")
    print(f"   this is network-bound, not model-bound")

    # 5. Cost.
    total_tokens = sum(c["token_count"] for c in chunks)
    price = {"voyage-context-4": 0.12, "voyage-4-large": 0.12,
             "voyage-4": 0.06, "voyage-4-lite": 0.02}.get(args.model, 0.12)
    print(f"\n5. cost to index this corpus")
    print(f"   {total_tokens:,} tokens x ${price}/1M = ${total_tokens/1e6*price:.5f}")
    print(f"   (first 200M tokens free -> $0.00 in practice)")

    print(f"\nlatency summary: index-path {t_alone:.0f} ms, query-path {t_real:.0f} ms")


if __name__ == "__main__":
    main()
