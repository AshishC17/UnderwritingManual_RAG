"""Qdrant collection schema and upsert.

The collection declares both a dense and a sparse vector from the start. Only the
dense one is populated here; the sparse slot is for BM25 in the hybrid-retrieval
block. Declaring it up front avoids migrating a populated collection later, and
Qdrant treats per-point sparse vectors as optional.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone

from src.guardrails.tracing import traceable
from qdrant_client import QdrantClient, models
from src.util.telemetry import local_tool
from src.store.index_context import collection_name

COLLECTION = "uw_manual"
DENSE = "dense"
SPARSE = "bm25"
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Payload indexes cover the fields used for lender/version scoping and provenance.
KEYWORD_INDEXES = [
    "chunk_type",
    "section",
    "subsection",
    "source_doc",
    "rule_codes_referenced",
    "lender_id",
    "product_id",
    "document_family",
    "document_id",
    "document_version",
    "version_status",
    "authority_level",
    "section_key",
    "subsection_key",
    "lineage_id",
    "chunking_version",
    "embedding_model",
    "embedding_version",
]
PAYLOAD_INDEXES = {
    **{field: models.PayloadSchemaType.KEYWORD for field in KEYWORD_INDEXES},
    "effective_from": models.PayloadSchemaType.DATETIME,
    "effective_to": models.PayloadSchemaType.DATETIME,
    "version_order": models.PayloadSchemaType.INTEGER,
    "chunk_revision": models.PayloadSchemaType.INTEGER,
    "has_footnote": models.PayloadSchemaType.BOOL,
}


def connect(url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=url)


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be uint or UUID; our chunk_ids are strings like
    `NovaCred_UW_V0_pdf.pdf::0033`, so derive a stable UUID from each."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def ensure_collection(
    client: QdrantClient, dims: int, recreate: bool = False, *, collection: str | None = None
) -> None:
    target = collection or collection_name()
    # Builds are immutable through the application APIs. Build lifecycle code
    # creates a fresh collection; never repurpose one behind a pinned turn.
    if target != COLLECTION and client.collection_exists(target):
        raise ValueError("versioned collection already exists; choose a new build")
    if target == COLLECTION and any(a.collection_name == target for a in client.get_aliases().aliases):
        raise ValueError("legacy collection is now aliased; use the candidate-build workflow")
    exists = client.collection_exists(target)
    if exists and recreate:
        client.delete_collection(target)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=target,
            vectors_config={
                DENSE: models.VectorParams(size=dims, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF is computed server-side; required for BM25-style scoring.
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    existing_indexes = set((client.get_collection(target).payload_schema or {}).keys())
    for field, schema in PAYLOAD_INDEXES.items():
        if field in existing_indexes:
            continue
        client.create_payload_index(
            collection_name=target,
            field_name=field,
            field_schema=schema,
        )


def _utc_datetime(value: str | date | datetime) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.combine(value, time.min)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def scope_filter(
    *,
    lender_id: str | None = None,
    document_id: str | None = None,
    document_version: str | None = None,
    current_only: bool = False,
    as_of: str | date | datetime | None = None,
) -> models.Filter | None:
    """Build an exact Qdrant filter after the resolver determines query scope."""
    if current_only and as_of is not None:
        raise ValueError("choose current_only or as_of, not both")

    must: list = []
    for key, value in (
        ("lender_id", lender_id),
        ("document_id", document_id),
        ("document_version", document_version),
    ):
        if value is not None:
            must.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    if current_only:
        must.append(
            models.FieldCondition(
                key="version_status", match=models.MatchValue(value="current")
            )
        )
    if as_of is not None:
        instant = _utc_datetime(as_of)
        must.extend(
            [
                models.FieldCondition(
                    key="effective_from",
                    range=models.DatetimeRange(lte=instant),
                ),
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="effective_to",
                            range=models.DatetimeRange(gte=instant),
                        ),
                        models.IsNullCondition(
                            is_null=models.PayloadField(key="effective_to")
                        ),
                        models.IsEmptyCondition(
                            is_empty=models.PayloadField(key="effective_to")
                        ),
                    ]
                ),
            ]
        )
    return models.Filter(must=must) if must else None


def upsert(
    client: QdrantClient,
    chunks: list[dict],
    vectors: list[list[float]],
    sparse: list[models.SparseVector] | None = None,
    *, collection: str | None = None,
) -> int:
    target = collection or collection_name()
    if target != COLLECTION or any(a.collection_name == target for a in client.get_aliases().aliases):
        raise ValueError("managed indexes are immutable; use the index build command")
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if sparse is not None and len(sparse) != len(chunks):
        raise ValueError(f"{len(chunks)} chunks but {len(sparse)} sparse vectors")

    points = []
    for i, (c, v) in enumerate(zip(chunks, vectors)):
        vec: dict = {DENSE: v}
        if sparse is not None:
            vec[SPARSE] = sparse[i]
        points.append(
            models.PointStruct(id=point_id(c["chunk_id"]), vector=vec, payload=c)
        )
    client.upsert(collection_name=target, points=points, wait=True)
    return len(points)


@local_tool("qdrant.get_by_chunk_ids")
def get_by_chunk_ids(client: QdrantClient, chunk_ids: list[str]) -> list[dict]:
    """Fetch chunk payloads by id, in the order given, skipping any not found.

    A direct point lookup, not a search — no vector, no scoring, no Voyage
    call. For re-hydrating chunks a previous turn already retrieved, this
    costs nothing but a local Qdrant read.
    """
    if not chunk_ids:
        return []
    points = client.retrieve(
        collection_name=collection_name(),
        ids=[point_id(cid) for cid in chunk_ids],
        with_payload=True,
    )
    by_id = {p.payload["chunk_id"]: p.payload for p in points}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


@traceable(run_type="retriever", name="search (dense)")
@local_tool("qdrant.search")
def search(
    client: QdrantClient,
    vector: list[float],
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """Dense-only search."""
    return client.query_points(
        collection_name=collection_name(),
        query=vector,
        using=DENSE,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points


@traceable(run_type="retriever", name="search_sparse (BM25)")
@local_tool("qdrant.search_sparse")
def search_sparse(
    client: QdrantClient,
    vector: models.SparseVector,
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """BM25-only search."""
    return client.query_points(
        collection_name=collection_name(),
        query=vector,
        using=SPARSE,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points


@traceable(run_type="retriever", name="search_hybrid (RRF)")
@local_tool("qdrant.search_hybrid")
def search_hybrid(
    client: QdrantClient,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    limit: int = 5,
    prefetch_limit: int = 20,
    query_filter: models.Filter | None = None,
    fusion: models.Fusion = models.Fusion.RRF,
) -> list[models.ScoredPoint]:
    """Run both retrievers and fuse them server-side.

    Fusion defaults to RRF, which combines by *rank* rather than score. Cosine
    similarity and BM25 scores live on unrelated scales — cosine is bounded, BM25
    is unbounded and shifts with corpus statistics — so averaging them directly
    would let whichever scale happens to be larger dominate. Ranks sidestep the
    problem and need no tuning.

    `prefetch_limit` is how deep each retriever goes before fusion. It must exceed
    `limit`: a document ranked 15th by one retriever and 2nd by the other should
    still be able to surface, which cannot happen if each only reports its top 5.
    """
    return client.query_points(
        collection_name=collection_name(),
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=DENSE,
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE,
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=fusion),
        limit=limit,
        with_payload=True,
    ).points
