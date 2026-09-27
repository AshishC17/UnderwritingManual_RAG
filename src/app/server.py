"""Chat server over the LangGraph pipeline.

One HTTP endpoint drives two different graph operations, and which one it is
depends on whether the thread is currently paused:

* No pending interrupt -> the message is a **question**. Start a fresh turn.
* Pending interrupt     -> the message is a **clarification reply**. Resume the
  paused node with it; the graph folds it into the question it clarifies.

That distinction is the whole state model, and it lives here rather than in the
client: the browser sends text and does not need to know which kind it is.

The service graph uses a PostgreSQL checkpointer.  Its state and the sanitised
user-visible transcript survive browser and service restarts.  Offline scripts
and evaluations still get an in-memory saver from ``build_graph()`` by default.
"""

from __future__ import annotations

import time
import uuid
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from src.graph.pipeline import build_graph  # noqa: E402
from src.graph.runtime import run_turn  # noqa: E402
from src.store import conversations as cv  # noqa: E402
from src.store import state_store  # noqa: E402
from src.eval.judge import JUDGE_MODEL  # noqa: E402
from src.resolve.scope import scope_key  # noqa: E402

CORRECTION_MODE = os.environ.get("RAG_CORRECTION_MODE", "post_generation")
DATABASE_URL = os.environ.get(
    "STATE_DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5432/rag"
)
STATIC = Path(__file__).parent / "static"
IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = ConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=int(os.environ.get("STATE_DB_POOL_SIZE", "10")),
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    pool.open(wait=True, timeout=20)
    try:
        state_store.assert_ready(pool)
        app.state.state_pool = pool
        # Opt in after the baseline alias has been bootstrapped. Resolve at the
        # start of each turn, never once at compile/startup time.
        index_selector = None
        if os.environ.get("RAG_INDEX_ALIAS"):
            from src.store.index_builds import resolve_active
            from src.store.qdrant_store import connect
            alias = os.environ["RAG_INDEX_ALIAS"]
            def index_selector():
                from contextlib import closing
                with closing(connect()) as client:
                    return resolve_active(client, alias)
        app.state.graph = build_graph(
            max_retries=1 if CORRECTION_MODE != "off" else 0,
            correction_mode=CORRECTION_MODE,
            review_model=os.environ.get("RAG_REVIEW_MODEL", JUDGE_MODEL),
            checkpointer=PostgresSaver(pool),
            index_selector=index_selector,
        )
        yield
    finally:
        pool.close()


app = FastAPI(title="Underwriting RAG", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # FastAPI's default error can echo the rejected raw input value.
    return JSONResponse(status_code=422, content={"detail": "Invalid request format or input length."})


class Message(BaseModel):
    thread_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=12000)


def current_principal(
    user_id: str = Header(default="learner-a", alias="X-Demo-User"),
) -> state_store.Principal:
    """Local Block-A identity seam; replace with verified SSO claims later."""
    if not IDENTIFIER.fullmatch(user_id):
        raise HTTPException(status_code=400, detail="Invalid local test identity.")
    return state_store.Principal(
        tenant_id=os.environ.get("LOCAL_DEMO_TENANT", "local-demo"),
        user_id=user_id,
    )


def _pool(request: Request):
    return request.app.state.state_pool


def _graph(request: Request):
    return request.app.state.graph


def _owned_thread(pool, principal: state_store.Principal, thread_id: str) -> dict:
    try:
        return state_store.get_thread(pool, principal, thread_id)
    except state_store.ThreadNotFound:
        # Do not reveal whether another user owns the identifier.
        raise HTTPException(status_code=404, detail="Conversation not found.") from None


@app.post("/threads")
def create_thread(
    request: Request,
    principal: state_store.Principal = Depends(current_principal),
) -> dict:
    return state_store.create_thread(_pool(request), principal)


@app.get("/threads")
def list_threads(
    request: Request,
    principal: state_store.Principal = Depends(current_principal),
) -> dict:
    return {"threads": state_store.list_threads(_pool(request), principal)}


@app.get("/threads/{thread_id}/messages")
def thread_messages(
    thread_id: str,
    request: Request,
    principal: state_store.Principal = Depends(current_principal),
) -> dict:
    if not IDENTIFIER.fullmatch(thread_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        thread, messages = state_store.list_messages(_pool(request), principal, thread_id)
    except state_store.ThreadNotFound:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None
    return {"thread": thread, "messages": messages}


@app.post("/chat")
def chat(
    body: Message,
    request: Request,
    principal: state_store.Principal = Depends(current_principal),
) -> dict:
    pool = _pool(request)
    if body.thread_id:
        thread = _owned_thread(pool, principal, body.thread_id)
    else:
        thread = state_store.create_thread(pool, principal, str(uuid.uuid4()))
    thread_id = thread["thread_id"]
    incoming_kind = (
        "clarification_reply" if thread["awaiting_clarification"] else "question"
    )
    started = time.monotonic()

    final = run_turn(_graph(request), thread_id, body.message)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    guard_conn = cv.connect()
    try:
        cv.record_guards(guard_conn, thread_id, final)
    finally:
        guard_conn.close()

    if final.get("guard_reason"):
        awaiting = bool(final.get("awaiting_clarification", False))
        response = {"thread_id": thread_id, "type": "answer", "text": final["answer"],
                    "awaiting_clarification": awaiting,
                    "elapsed_ms": elapsed_ms, "guardrails": final["guard_events"],
                    "guard_usage": final["guard_usage"],
                    "request_usage": final["request_usage"],
                    "correction": {"status": final["final_status"], "attempts": 0}}
        # Rejected raw input is deliberately absent from the durable transcript.
        state_store.append_exchange(
            pool, principal, thread_id,
            [{"role": "assistant", "kind": "blocked", "content": final["answer"],
              "metadata": {"guard_reason": final.get("guard_reason")}}],
            awaiting_clarification=awaiting,
        )
        return response

    interrupts = final.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        response = {
            "thread_id": thread_id,
            "type": "clarification",
            "text": payload["question"],
            "reason": payload.get("reason"),
            "elapsed_ms": elapsed_ms,
            "usage": final.get("usage_summary"),
            "guardrails": final.get("guard_events", []),
            "guard_usage": final.get("guard_usage"),
            "request_usage": final.get("request_usage"),
            "correction": {"status": "clarification", "attempts": final.get("retry_attempt", 0)},
        }
        state_store.append_exchange(
            pool, principal, thread_id,
            [
                {"role": "user", "kind": incoming_kind, "content": body.message},
                {"role": "assistant", "kind": "clarification_prompt",
                 "content": payload["question"], "metadata": {"reason": payload.get("reason")}},
            ],
            awaiting_clarification=True,
        )
        return response

    # Only a completed answer is a turn. A clarification is not a question of
    # its own, so it never produces a row.
    return _finalize_answer_turn(
        pool, principal, thread_id, final, elapsed_ms,
        incoming_kind=incoming_kind, raw_message=body.message,
    )


def _answer_response(thread_id: str, turn_id, final: dict, elapsed_ms: int, link) -> dict:
    scope = final.get("scope") or {}
    comparison_scopes = [
        {
            "key": scope_key(item),
            "lender": item.get("lender_name"),
            "version": item.get("document_version"),
            "status": item.get("status"),
        }
        for item in (final.get("comparison_scopes") or [])
    ]
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "type": "answer",
        "text": final.get("answer") or "",
        "resolved_question": final.get("resolved_question"),
        "scope": {
            "lender": scope.get("lender_name"),
            "version": scope.get("document_version"),
            "status": scope.get("status"),
            "inherited": bool(final.get("scope_inherited")),
        },
        "scopes": comparison_scopes,
        "turn_number": len(final.get("history") or []) + 1,
        "sub_queries": final.get("sub_queries", []),
        "comparison_sub_queries": final.get("comparison_sub_queries", {}),
        "ground_truth": (
            {
                "case_id": link["case_id"],
                "comparable": bool(link["scope_comparable"]),
                "note": link["scope_note"],
            }
            if link
            else None
        ),
        "elapsed_ms": elapsed_ms,
        "usage": final.get("usage_summary"),
        "guardrails": final.get("guard_events", []),
        "guard_usage": final.get("guard_usage"),
        "request_usage": final.get("request_usage"),
        "correction": {"mode": final.get("correction_mode"), "status": final.get("final_status"),
                       "attempts": final.get("retry_attempt", 0), "review_model": final.get("review_model"),
                       "decisions": [{"attempt": r["attempt"], "action": r["selected_action"], "reason": r["review"].get("reason")}
                                     for r in final.get("review_log", [])]},
    }


def _finalize_answer_turn(
    pool, principal: state_store.Principal, thread_id: str, final: dict, elapsed_ms: int,
    *, incoming_kind: str, raw_message: str,
) -> dict:
    """Score against ground truth (SQLite, unchanged) and durably commit the
    turn (Postgres, idempotent by `turn_id`).

    Called from both the normal `/chat` path and `/recover`. `raw_message`
    is the literal HTTP body for a fresh `/chat` call; for a `/recover` call
    on a resumed thread it is best-effort `final["question"]`, which is the
    turn's original opening question, not necessarily the exact clarification
    reply text -- that raw text is not preserved anywhere recoverable once
    LangGraph has folded it into the resumed node's own state. Documented
    honestly rather than silently guessed at.
    """
    conn = cv.connect()
    try:
        turn_id_sqlite = cv.record_turn(
            conn, thread_id=thread_id, raw_question=final["question"],
            state=final, latency_ms=elapsed_ms,
        )
        link = conn.execute(
            "SELECT case_id, scope_comparable, scope_note FROM gt_links WHERE turn_id = ?",
            (turn_id_sqlite,),
        ).fetchone()
    finally:
        conn.close()

    response = _answer_response(thread_id, turn_id_sqlite, final, elapsed_ms, link)
    response["index_build"] = {key: final.get("index_binding", {}).get(key)
                               for key in ("build_id", "collection", "manifest_sha256")}
    turn_id = final.get("turn_id")
    if turn_id:
        commit = state_store.commit_turn(
            pool, principal, thread_id,
            turn_id=turn_id, graph_version=final.get("graph_version", "unknown"),
            question=final["question"], answer=response["text"],
            scope=response["scope"], evidence_chunk_ids=[
                c["chunk_id"] for c in (final.get("context_chunks") or [])
            ],
            messages=[
                {"role": "user", "kind": incoming_kind, "content": raw_message},
                {"role": "assistant", "kind": "answer", "content": response["text"],
                 "metadata": {"turn_id": turn_id_sqlite, "scope": response["scope"],
                              "scopes": response["scopes"], "index_build": response["index_build"]}},
            ],
        )
        response["recovery"] = {"turn_id": turn_id, "already_committed": commit["already_existed"]}
    return response


@app.get("/threads/{thread_id}/recover")
def recover(
    thread_id: str,
    request: Request,
    principal: state_store.Principal = Depends(current_principal),
) -> dict:
    """Finish a durable commit that a crash interrupted -- never re-invokes
    the graph, never repeats a provider call. Only reads the checkpoint
    already sitting in PostgresSaver and, if it looks like a completed turn,
    idempotently commits it via the same path `/chat` uses.

    Deliberately does not attempt automatic recovery for a thread that died
    mid-node, no `__interrupt__`): resuming that blindly could replay a
    provider call or release an unreviewed draft. The caller is told recovery
    is unavailable and should ask the question again.

    "Died mid-node" is detected from `snapshot.next` -- LangGraph's own record
    of which node(s) still have not run -- not from `final_status`. That field
    looked tempting (`"not_reviewed"` reads like "done, nothing to review")
    but `begin_turn_node` also writes `"not_reviewed"` as its reset default at
    the *start* of every turn, before retrieval or generation ever runs. A
    real `kill -9` inside `retrieve_node` proved the bug this caused: the
    checkpoint's `final_status` was still `"not_reviewed"` from that reset,
    so this endpoint treated an empty, never-generated answer as a completed
    turn and durably committed it. `snapshot.next` has no such ambiguity --
    it is empty only once the graph has actually reached `END`.
    """
    pool = _pool(request)
    thread = _owned_thread(pool, principal, thread_id)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if any(task.interrupts for task in snapshot.tasks):
        return {"thread_id": thread_id, "status": "awaiting_clarification",
                "detail": "Thread is paused on a clarification reply; nothing to recover."}

    final = snapshot.values
    if not final.get("turn_id") or snapshot.next:
        return {"thread_id": thread_id, "status": "recovery_unavailable",
                "detail": "No completed turn found in this checkpoint. Ask the question again."}

    response = _finalize_answer_turn(
        pool, principal, thread_id, final, elapsed_ms=0,
        incoming_kind="question", raw_message=final.get("question", ""),
    )
    response["status"] = "recovered"
    return response


@app.get("/coverage")
def coverage() -> dict:
    """How much of the logged conversation is scorable against ground truth."""
    conn = cv.connect()
    try:
        return cv.coverage(conn)
    finally:
        conn.close()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
