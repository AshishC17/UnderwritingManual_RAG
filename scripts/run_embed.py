"""Embed chunks and load them into Qdrant.

    python scripts/run_embed.py [--recreate]

Requires VOYAGE_API_KEY. Embeddings are cached on disk, so re-running after the
first successful pass costs nothing and hits no API.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.embed.embedder import DIMS, MODEL, MissingCredentials, embed_document
from src.embed.sparse import embed_documents as embed_sparse
from src.store import qdrant_store as qs

# Reads VOYAGE_API_KEY from .env at the project root (gitignored).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_CHUNKS = "data/processed/chunks_v2.json"


def embed_by_document(
    chunks: list[dict], *, model: str, dims: int
) -> list[list[float]]:
    """Contextualize each source document independently and restore corpus order.

    Passing all four manuals in one Voyage contextual-embedding call would allow
    context from one lender/version to influence another lender/version's vectors.
    """
    indexes_by_source: dict[str, list[int]] = defaultdict(list)
    from src.guardrails.checks import evidence_check
    evidence_check(chunks)
    for index, item in enumerate(chunks):
        indexes_by_source[item["source_doc"]].append(index)

    vectors: list[list[float]] = [[] for _ in chunks]
    for source_doc, indexes in indexes_by_source.items():
        texts = [chunks[index]["text"] for index in indexes]
        embedded = embed_document(texts, model=model, dims=dims)
        if len(embedded) != len(indexes):
            raise ValueError(
                f"{source_doc}: expected {len(indexes)} vectors, got {len(embedded)}"
            )
        for index, vector in zip(indexes, embedded):
            vectors[index] = vector
        print(f"  contextual group: {source_doc} ({len(indexes)} chunks)")
    return vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dims", type=int, default=DIMS)
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the collection (use after a chunking change)",
    )
    args = ap.parse_args()

    chunks = [dict(item) for item in json.loads(Path(args.chunks).read_text())]
    embedding_version = f"{args.model}:{args.dims}:document-context"
    for item in chunks:
        item["embedding_model"] = args.model
        item["embedding_dimension"] = args.dims
        item["embedding_version"] = embedding_version
    texts = [c["text"] for c in chunks]
    print(f"chunks: {len(chunks)}  model: {args.model}  dims: {args.dims}")

    try:
        vectors = embed_by_document(chunks, model=args.model, dims=args.dims)
    except MissingCredentials as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)

    if len(vectors) != len(chunks):
        raise SystemExit(
            f"embedding returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    print(f"embedded: {len(vectors)} dense vectors of dim {len(vectors[0])}")

    # BM25 runs locally — no API, no rate limit, no cost.
    sparse = embed_sparse(texts)
    nnz = sum(len(s.indices) for s in sparse) / len(sparse)
    print(f"embedded: {len(sparse)} sparse vectors, {nnz:.0f} avg terms each")

    client = qs.connect(args.qdrant)
    qs.ensure_collection(client, dims=args.dims, recreate=args.recreate)
    n = qs.upsert(client, chunks, vectors, sparse=sparse)
    print(f"upserted: {n} points -> {qs.COLLECTION}")
    print(f"collection now holds: {client.get_collection(qs.COLLECTION).points_count}")


if __name__ == "__main__":
    main()
