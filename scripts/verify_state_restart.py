#!/usr/bin/env python3
"""Deterministic Block-A verification against the local PostgreSQL service.

No model, embedding, reranking, Qdrant, or LangSmith call leaves the process.
The script starts two application lifespans for each case to represent a real
service restart while preserving PostgreSQL.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from fastapi.testclient import TestClient

from src.app import server
from src.resolve.scope import resolve_query_scope


QUESTION = "For NovaCred V1 explain the Blocked SSN rule."
SCOPE = resolve_query_scope(QUESTION)
CHUNK = {
    "chunk_id": "durable-demo::0001",
    "text": "A Blocked SSN requires review.",
    "lender_id": SCOPE["lender_id"],
    "document_version": SCOPE["document_version"],
    "document_id": SCOPE["document_id"],
}


def main() -> None:
    suffix = uuid.uuid4().hex[:8]
    pending_headers = {"X-Demo-User": f"pending-{suffix}"}
    completed_headers = {"X-Demo-User": f"completed-{suffix}"}
    foreign_headers = {"X-Demo-User": f"foreign-{suffix}"}
    server.CORRECTION_MODE = "off"

    patches = [
        patch("src.graph.runtime.request_check", side_effect=lambda text: text),
        patch("src.graph.runtime.tracing_is_enabled", return_value=False),
        patch("src.graph.pipeline.classify_followup", side_effect=lambda q, h, **kw: {
            "behavior": "answer_policy_question", "standalone": q, "instruction": q,
        }),
        patch("src.graph.pipeline.decompose_query", side_effect=lambda q: [q]),
        patch("src.graph.pipeline.qs.connect", return_value=object()),
        patch("src.graph.pipeline.embed_query_scoped", return_value=[0.0]),
        patch("src.graph.pipeline.embed_query_sparse", return_value=object()),
        patch("src.graph.pipeline.qs.search_hybrid", return_value=[
            SimpleNamespace(payload=CHUNK, score=1)
        ]),
        patch("src.graph.pipeline.rerank", side_effect=lambda q, chunks, **kw: [
            (chunk, 1) for chunk in chunks
        ]),
        patch("src.graph.pipeline.generate", return_value=(
            "A Blocked SSN requires review. [durable-demo::0001]"
        )),
        patch("src.graph.pipeline.evidence_check", return_value=None),
    ]
    for active_patch in patches:
        active_patch.start()

    try:
        with TestClient(server.app) as client:
            first = client.post(
                "/chat", headers=pending_headers,
                json={"message": "What happens in the underwriting manual?"},
            )
            first.raise_for_status()
            assert first.json()["type"] == "clarification"
            pending_thread = first.json()["thread_id"]
            assert client.get(
                f"/threads/{pending_thread}/messages", headers=foreign_headers
            ).status_code == 404

        with TestClient(server.app) as client:
            restored = client.get(
                f"/threads/{pending_thread}/messages", headers=pending_headers
            )
            restored.raise_for_status()
            assert restored.json()["thread"]["awaiting_clarification"] is True
            resumed = client.post(
                "/chat", headers=pending_headers,
                json={
                    "thread_id": pending_thread,
                    "message": "I still do not know the lender.",
                },
            )
            resumed.raise_for_status()
            assert resumed.json()["type"] == "clarification"

        with TestClient(server.app) as client:
            completed = client.post(
                "/chat", headers=completed_headers, json={"message": QUESTION}
            )
            completed.raise_for_status()
            assert completed.json()["type"] == "answer"
            completed_thread = completed.json()["thread_id"]

        with TestClient(server.app) as client:
            transcript = client.get(
                f"/threads/{completed_thread}/messages", headers=completed_headers
            )
            transcript.raise_for_status()
            state = client.app.state.graph.get_state({
                "configurable": {"thread_id": completed_thread}
            }).values
            assert [item["kind"] for item in transcript.json()["messages"]] == [
                "question", "answer"
            ]
            assert len(state.get("history") or []) == 1
            assert state["scope"]["lender_id"] == "novacred"
            assert state["scope"]["document_version"] == "V1"

        print("Block A restart verification passed")
        print(f"pending_thread={pending_thread}")
        print(f"completed_thread={completed_thread}")
        print("ownership_isolation=passed")
        print("pending_interrupt_resume=passed")
        print("completed_history_restore=passed")
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


if __name__ == "__main__":
    main()
