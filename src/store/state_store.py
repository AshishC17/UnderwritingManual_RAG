"""PostgreSQL records that make conversations reopenable and owner-scoped.

LangGraph's PostgresSaver owns execution checkpoints.  These tables own the
application view of a conversation: who may open it and which sanitised
messages should be rendered after a browser or service restart.

Block B's request ledger, cross-thread locks and revision checks are still
not here. Block C adds one narrower thing on top of Block A: a durable,
idempotent record of each completed turn's *result*, keyed by a `turn_id`
that lives inside the LangGraph checkpoint itself. Without it, a crash
between "the graph finished" and "the transcript was written" leaves the
checkpoint believing the turn is done while `rag_app.messages` never
recorded it -- and a naive retry would then produce a duplicate turn. A
`turn_id`-keyed idempotent insert is what makes `commit_turn` safe to call
twice, whether from `/chat` itself or from `/recover` after a restart.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Iterable

from psycopg.types.json import Jsonb

from src.guardrails.privacy import redact, sanitize


MIGRATION_VERSION = 2

MIGRATION_SQL = """
CREATE SCHEMA IF NOT EXISTS rag_app;

CREATE TABLE IF NOT EXISTS rag_app.schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_app.threads (
    thread_id              TEXT PRIMARY KEY,
    tenant_id              TEXT NOT NULL,
    user_id                TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'active'
                           CHECK (status IN ('active', 'awaiting_clarification', 'closed')),
    awaiting_clarification BOOLEAN NOT NULL DEFAULT FALSE,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_threads_owner_updated
    ON rag_app.threads (tenant_id, user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS rag_app.messages (
    message_seq BIGSERIAL PRIMARY KEY,
    message_id  UUID NOT NULL UNIQUE,
    thread_id   TEXT NOT NULL REFERENCES rag_app.threads(thread_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    kind        TEXT NOT NULL CHECK (kind IN ('question', 'clarification_reply',
                                              'clarification_prompt', 'answer',
                                              'blocked', 'system')),
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_messages_thread_sequence
    ON rag_app.messages (thread_id, message_seq);

CREATE TABLE IF NOT EXISTS rag_app.turns (
    turn_id            UUID PRIMARY KEY,
    thread_id          TEXT NOT NULL REFERENCES rag_app.threads(thread_id) ON DELETE CASCADE,
    graph_version      TEXT NOT NULL,
    question           TEXT NOT NULL,
    answer             TEXT NOT NULL,
    scope              JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_hash      TEXT NOT NULL,
    committed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_turns_thread
    ON rag_app.turns (thread_id, committed_at);
"""


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str


class ThreadNotFound(LookupError):
    """The thread does not exist or is not owned by this principal."""


def migrate(pool) -> None:
    """Create Block-A application tables. Run from the setup command only."""
    with pool.connection() as conn, conn.transaction():
        # The migration is intentionally one reviewed SQL block.  Psycopg's
        # prepared-statement protocol accepts one statement at a time, so DDL
        # setup uses the simple protocol explicitly.
        conn.execute(MIGRATION_SQL, prepare=False)
        conn.execute(
            "INSERT INTO rag_app.schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
            (MIGRATION_VERSION,),
        )


def assert_ready(pool) -> None:
    """Fail startup clearly rather than falling back to process memory."""
    with pool.connection() as conn:
        row = conn.execute(
            """SELECT to_regclass('rag_app.schema_migrations') AS app_table,
                      to_regclass('public.checkpoints') AS checkpoint_table"""
        ).fetchone()
        app_ready = bool(row and row["app_table"])
        checkpoints_ready = bool(row and row["checkpoint_table"])
        if app_ready:
            app_ready = bool(conn.execute(
                "SELECT 1 FROM rag_app.schema_migrations WHERE version = %s",
                (MIGRATION_VERSION,),
            ).fetchone())
    if not app_ready or not checkpoints_ready:
        raise RuntimeError(
            "PostgreSQL state schema is not initialized; run "
            "`.venv/bin/python scripts/setup_state_db.py` first."
        )


def create_thread(pool, principal: Principal, thread_id: str | None = None) -> dict:
    thread_id = thread_id or str(uuid.uuid4())
    with pool.connection() as conn:
        row = conn.execute(
            """INSERT INTO rag_app.threads (thread_id, tenant_id, user_id)
               VALUES (%s, %s, %s)
               RETURNING thread_id, status, awaiting_clarification, created_at, updated_at""",
            (thread_id, principal.tenant_id, principal.user_id),
        ).fetchone()
    return _thread_dict(row)


def get_thread(pool, principal: Principal, thread_id: str) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            """SELECT thread_id, status, awaiting_clarification, created_at, updated_at
               FROM rag_app.threads
               WHERE thread_id = %s AND tenant_id = %s AND user_id = %s""",
            (thread_id, principal.tenant_id, principal.user_id),
        ).fetchone()
    if row is None:
        raise ThreadNotFound(thread_id)
    return _thread_dict(row)


def list_threads(pool, principal: Principal, limit: int = 50) -> list[dict]:
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT thread_id, status, awaiting_clarification, created_at, updated_at
               FROM rag_app.threads
               WHERE tenant_id = %s AND user_id = %s
               ORDER BY updated_at DESC
               LIMIT %s""",
            (principal.tenant_id, principal.user_id, limit),
        ).fetchall()
    return [_thread_dict(row) for row in rows]


def list_messages(pool, principal: Principal, thread_id: str) -> tuple[dict, list[dict]]:
    thread = get_thread(pool, principal, thread_id)
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT message_id, role, kind, content, metadata, created_at
               FROM rag_app.messages
               WHERE thread_id = %s
               ORDER BY message_seq""",
            (thread_id,),
        ).fetchall()
    return thread, [
        {
            "message_id": str(row["message_id"]),
            "role": row["role"],
            "kind": row["kind"],
            "content": row["content"],
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


def append_exchange(
    pool,
    principal: Principal,
    thread_id: str,
    messages: Iterable[dict],
    *,
    awaiting_clarification: bool,
) -> None:
    """Persist one user-visible exchange and update the thread atomically."""
    items = list(messages)
    with pool.connection() as conn, conn.transaction():
        owner = conn.execute(
            """SELECT 1 FROM rag_app.threads
               WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
               FOR UPDATE""",
            (thread_id, principal.tenant_id, principal.user_id),
        ).fetchone()
        if owner is None:
            raise ThreadNotFound(thread_id)

        for item in items:
            content = redact(str(item["content"]), "transcript_write")
            metadata = sanitize(item.get("metadata") or {}, "transcript_write")
            conn.execute(
                """INSERT INTO rag_app.messages
                       (message_id, thread_id, role, kind, content, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    uuid.uuid4(), thread_id, item["role"], item["kind"],
                    content, Jsonb(metadata),
                ),
            )

        conn.execute(
            """UPDATE rag_app.threads
               SET status = %s, awaiting_clarification = %s, updated_at = now()
               WHERE thread_id = %s""",
            (
                "awaiting_clarification" if awaiting_clarification else "active",
                awaiting_clarification,
                thread_id,
            ),
        )


def evidence_hash(chunk_ids: Iterable[str]) -> str:
    """Fingerprint of *which chunks* a turn's answer was built from.

    Covers chunk-set identity, not byte-identical text under those ids --
    proving text hasn't drifted under a stable id would need a full-text
    snapshot per turn, which this first pass does not add. This still
    catches the case this project has hit before: a re-chunk or re-embed
    that changes what a stored chunk_id even points at.
    """
    ordered = "|".join(sorted(chunk_ids))
    return hashlib.sha256(ordered.encode()).hexdigest()[:24]


def commit_turn(
    pool,
    principal: Principal,
    thread_id: str,
    *,
    turn_id: str,
    graph_version: str,
    question: str,
    answer: str,
    scope: dict,
    evidence_chunk_ids: list[str],
    messages: Iterable[dict],
) -> dict:
    """Durably record one completed turn's result, exactly once.

    Safe to call twice with the same `turn_id` -- from `/chat` itself, or
    from `/recover` after a restart finds a checkpoint whose transcript
    commit never happened. The second call is a no-op: it does not
    re-insert the transcript messages, since the first call already did.
    """
    content_hash = evidence_hash(evidence_chunk_ids)
    items = list(messages)
    with pool.connection() as conn, conn.transaction():
        owner = conn.execute(
            """SELECT 1 FROM rag_app.threads
               WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
               FOR UPDATE""",
            (thread_id, principal.tenant_id, principal.user_id),
        ).fetchone()
        if owner is None:
            raise ThreadNotFound(thread_id)

        inserted = conn.execute(
            """INSERT INTO rag_app.turns
                   (turn_id, thread_id, graph_version, question, answer,
                    scope, evidence_chunk_ids, evidence_hash)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (turn_id) DO NOTHING
               RETURNING turn_id""",
            (
                turn_id, thread_id, graph_version,
                redact(question, "transcript_write"), redact(answer, "transcript_write"),
                Jsonb(sanitize(scope, "transcript_write")), Jsonb(evidence_chunk_ids), content_hash,
            ),
        ).fetchone()
        if inserted is None:
            return {"committed": False, "already_existed": True, "evidence_hash": content_hash}

        for item in items:
            content = redact(str(item["content"]), "transcript_write")
            metadata = sanitize(item.get("metadata") or {}, "transcript_write")
            conn.execute(
                """INSERT INTO rag_app.messages
                       (message_id, thread_id, role, kind, content, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (uuid.uuid4(), thread_id, item["role"], item["kind"], content, Jsonb(metadata)),
            )
        conn.execute(
            """UPDATE rag_app.threads SET status = 'active', awaiting_clarification = FALSE,
               updated_at = now() WHERE thread_id = %s""",
            (thread_id,),
        )
    return {"committed": True, "already_existed": False, "evidence_hash": content_hash}


def _thread_dict(row) -> dict:
    return {
        "thread_id": row["thread_id"],
        "status": row["status"],
        "awaiting_clarification": bool(row["awaiting_clarification"]),
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }
