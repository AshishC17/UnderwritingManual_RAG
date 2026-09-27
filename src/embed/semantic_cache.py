"""Scope-partitioned semantic caching in front of `embed_query`.

The exact-hash cache in `embedder.py` only matches byte-identical text. This
sits in front of it: on an exact miss, it checks whether a *similar* query has
been embedded before, within the same lender+version, and if so reuses that
query's real Voyage vector instead of paying for a new embedding call.

Two embedders are in play, doing different jobs. Using Voyage to decide
whether to call Voyage would save nothing — same API call, same rate-limit
wait either way. So the similarity check runs on a small local model
(FastEmbed, already a project dependency for BM25) that costs nothing and has
no rate limit. Voyage stays the only thing that ever produces the vector
actually used for ranking; the local model only ever decides "have I seen
something like this before."

Scope partitioning is the safety property, not an add-on. A query resolved to
NovaCred V1 is only ever compared against other NovaCred V1 entries — V0 and
LumenTrail are structurally unreachable here, regardless of how close two
vectors land. That is what makes this safe to add after the "Code 128 means
something different in V0 vs V1" failure mode: partitioning removes the
dangerous case by construction, rather than trying to catch it after the fact.

No verification step gates a hit before it's used — every decision is logged
instead (query, scope, matched query, similarity, and the best candidate seen
even on a miss) so hit quality can be hand-audited later, the same protocol
already used to validate the LLM judge. The blast radius of a wrong hit is
bounded: it can only skew ranking within the correct, already-filtered scope,
never blend evidence across lenders or versions.

The similarity threshold below is a placeholder, not a measured value — there
is no calibration data yet. It is deliberately conservative (few false hits,
more true misses) until the audit log has enough real decisions to tune it
against.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.embed.embedder import CACHE_DIR as VOYAGE_CACHE_DIR
from src.embed.embedder import DIMS as VOYAGE_DIMS
from src.embed.embedder import MODEL as VOYAGE_MODEL
from src.embed.embedder import _is_cached as _voyage_is_cached
from src.embed.embedder import embed_query

DB_PATH = "data/semantic_cache.db"
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

# Placeholder pending calibration against real logged decisions — see module
# docstring. Deliberately high: a missed reuse costs a Voyage call and a
# rate-limit wait; a wrong reuse skews an answer, so the starting bias favors
# missing over guessing.
DEFAULT_THRESHOLD = 0.92

SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_queries (
    id           TEXT PRIMARY KEY,
    scope_key    TEXT NOT NULL,
    query_text   TEXT NOT NULL,
    local_vector TEXT NOT NULL,
    local_model  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semantic_queries_scope ON semantic_queries (scope_key);

CREATE TABLE IF NOT EXISTS semantic_cache_log (
    id              TEXT PRIMARY KEY,
    scope_key       TEXT NOT NULL,
    query_text      TEXT NOT NULL,
    decision        TEXT NOT NULL CHECK (decision IN ('exact_hit', 'semantic_hit', 'miss')),
    matched_query   TEXT,
    similarity      REAL,
    threshold       REAL NOT NULL,
    best_candidate  TEXT,
    best_similarity REAL,
    created_at      TEXT NOT NULL
);
"""


@dataclass
class ScopeInput:
    lender_id: str
    document_version: str


def scope_key(lender_id: str, document_version: str) -> str:
    """The partition boundary: same lender AND same version, nothing looser.

    Deliberately not just `lender_id` — NovaCred V0 and V1 must be as isolated
    from each other as NovaCred is from LumenTrail, since the failure that
    motivated this design was a same-lender, different-version collision.
    """
    return f"{lender_id}::{document_version}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    db = Path(path)
    if db.parent != Path("."):
        db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


_local_model = None


def _local_embed(text: str) -> np.ndarray:
    """Embed with the small local model. Lazily loaded once per process —
    FastEmbed reads model weights from disk on first use, not worth repeating
    per call."""
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding

        _local_model = TextEmbedding(model_name=LOCAL_MODEL)
    return next(_local_model.embed([text]))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _best_match(
    conn: sqlite3.Connection, key: str, vector: np.ndarray
) -> tuple[str, float] | None:
    """Highest-similarity past query in this scope partition, or None if the
    partition is empty. Brute-force: at this project's real scale (tens to
    low hundreds of distinct queries) a linear scan costs microseconds, and a
    real index would be machinery this doesn't need yet."""
    rows = conn.execute(
        "SELECT query_text, local_vector FROM semantic_queries WHERE scope_key = ?",
        (key,),
    ).fetchall()
    if not rows:
        return None
    best_text, best_score = None, -1.0
    for row in rows:
        candidate = np.array(json.loads(row["local_vector"]))
        score = _cosine(vector, candidate)
        if score > best_score:
            best_text, best_score = row["query_text"], score
    return best_text, best_score


def _record_query(conn: sqlite3.Connection, key: str, text: str, vector: np.ndarray) -> None:
    conn.execute(
        "INSERT INTO semantic_queries (id, scope_key, query_text, local_vector, local_model, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), key, text, json.dumps(vector.tolist()), LOCAL_MODEL, _now()),
    )


def _log(conn: sqlite3.Connection, **fields) -> None:
    fields.setdefault("matched_query", None)
    fields.setdefault("similarity", None)
    fields.setdefault("best_candidate", None)
    fields.setdefault("best_similarity", None)
    conn.execute(
        "INSERT INTO semantic_cache_log "
        "(id, scope_key, query_text, decision, matched_query, similarity, "
        " threshold, best_candidate, best_similarity, created_at) "
        "VALUES (:id,:scope_key,:query_text,:decision,:matched_query,:similarity,"
        " :threshold,:best_candidate,:best_similarity,:created_at)",
        {"id": str(uuid.uuid4()), "created_at": _now(), **fields},
    )


def embed_query_scoped(
    text: str,
    lender_id: str,
    document_version: str,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = VOYAGE_MODEL,
    dims: int = VOYAGE_DIMS,
    voyage_cache_dir: str = VOYAGE_CACHE_DIR,
    db_path: str = DB_PATH,
) -> list[float]:
    """Return a query's Voyage embedding, reusing a scoped near-duplicate's
    vector when the exact-hash cache misses and a close match exists.

    Every call is logged with its decision, so hit quality can be audited
    without re-running anything — see the module docstring for why no
    synchronous verification gates a hit before it's used.
    """
    from src.guardrails.privacy import redact
    from src.guardrails.core import cache_directory
    from src.store.index_context import current
    binding = current()
    model = binding.get("dense", {}).get("model", model)
    dims = binding.get("dense", {}).get("dims", dims)
    text = redact(text, "semantic_cache_input")
    voyage_cache_dir = cache_directory(voyage_cache_dir)
    if binding.get("disable_semantic_cache"):
        return embed_query(text, model, dims, voyage_cache_dir)
    key = scope_key(lender_id, document_version)
    if binding:
        key += f"::{binding['build_id']}::{model}::{dims}::{LOCAL_MODEL}"
    conn = connect(db_path)
    try:
        if _voyage_is_cached([text], model, dims, "query", voyage_cache_dir):
            _log(conn, scope_key=key, query_text=text, decision="exact_hit", threshold=threshold)
            conn.commit()
            return embed_query(text, model, dims, voyage_cache_dir)

        local_vector = _local_embed(text)
        match = _best_match(conn, key, local_vector)

        if match and match[1] >= threshold:
            matched_text, similarity = match
            _log(
                conn, scope_key=key, query_text=text, decision="semantic_hit",
                matched_query=matched_text, similarity=similarity, threshold=threshold,
                best_candidate=matched_text, best_similarity=similarity,
            )
            _record_query(conn, key, text, local_vector)
            conn.commit()
            # The matched query was for-real embedded before, so this resolves
            # to an exact-hash hit in embedder.py's own cache — no new Voyage
            # call, no new rate-limit wait.
            return embed_query(matched_text, model, dims, voyage_cache_dir)

        _log(
            conn, scope_key=key, query_text=text, decision="miss", threshold=threshold,
            best_candidate=match[0] if match else None,
            best_similarity=match[1] if match else None,
        )
        _record_query(conn, key, text, local_vector)
        conn.commit()
        return embed_query(text, model, dims, voyage_cache_dir)
    finally:
        conn.close()


def audit_summary(db_path: str = DB_PATH) -> dict:
    """Decision counts and the similarity spread around the threshold — the
    numbers a hand-audit pass would start from, not a substitute for one."""
    conn = connect(db_path)
    try:
        counts = {
            row["decision"]: row["n"]
            for row in conn.execute(
                "SELECT decision, COUNT(*) AS n FROM semantic_cache_log GROUP BY decision"
            )
        }
        near_misses = conn.execute(
            "SELECT query_text, best_candidate, best_similarity, threshold "
            "FROM semantic_cache_log WHERE decision = 'miss' AND best_similarity IS NOT NULL "
            "ORDER BY best_similarity DESC LIMIT 20"
        ).fetchall()
        return {
            "total": sum(counts.values()),
            "by_decision": counts,
            "closest_misses": [dict(r) for r in near_misses],
        }
    finally:
        conn.close()
