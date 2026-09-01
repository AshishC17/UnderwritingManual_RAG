"""Qdrant collection schema and upsert.

The collection declares both a dense and a sparse vector from the start. Only the
dense one is populated here; the sparse slot is for BM25 in the hybrid-retrieval
block. Declaring it up front avoids migrating a populated collection later, and
Qdrant treats per-point sparse vectors as optional.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

COLLECTION = "uw_manual"
DENSE = "dense"
SPARSE = "bm25"
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Payload fields worth indexing: the ones we expect to filter on.
KEYWORD_INDEXES = ["chunk_type", "section", "source_doc", "rule_codes_referenced"]


def connect(url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=url)


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be uint or UUID; our chunk_ids are strings like
    `NovaCred_UW_V0_pdf.pdf::0033`, so derive a stable UUID from each."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def ensure_collection(
    client: QdrantClient, dims: int, recreate: bool = False
) -> None:
    exists = client.collection_exists(COLLECTION)
    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False
    if exists:
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE: models.VectorParams(size=dims, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            # IDF is computed server-side; required for BM25-style scoring.
            SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    for field in KEYWORD_INDEXES:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def upsert(
    client: QdrantClient, chunks: list[dict], vectors: list[list[float]]
) -> int:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    points = [
        models.PointStruct(
            id=point_id(c["chunk_id"]),
            vector={DENSE: v},
            payload=c,
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    return len(points)


def search(
    client: QdrantClient,
    vector: list[float],
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    return client.query_points(
        collection_name=COLLECTION,
        query=vector,
        using=DENSE,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points
