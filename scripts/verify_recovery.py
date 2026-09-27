#!/usr/bin/env python3
"""Block C kill-point verification against the local PostgreSQL service.

No model, embedding, reranking, Qdrant, or LangSmith call leaves the process.

Covers kill-point 3 (graph finished, transcript commit never ran) and
kill-point 4 (already committed, /recover called again -- must be a no-op),
plus the mid-node-death case (no auto-resume, an honest "ask again").

Kill-points 1 and 2 (mid-retrieval, mid-generation) need a real `kill -9` on
a running server process -- a TestClient cannot simulate that truthfully, so
this script does not claim to cover them. That gap is real, not hidden.
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
from src.graph.runtime import run_turn
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
    headers = {"X-Demo-User": f"recovery-{suffix}"}
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
            created = client.post("/threads", headers=headers)
            created.raise_for_status()
            thread_id = created.json()["thread_id"]

            # --- mid-node death: nothing has run on this thread at all ---
            never_run = client.get(f"/threads/{thread_id}/recover", headers=headers)
            never_run.raise_for_status()
            assert never_run.json()["status"] == "recovery_unavailable", never_run.json()

            # --- kill-point 3: run the graph directly, bypassing /chat, so it
            # never reaches _finalize_answer_turn -- the checkpoint ends up
            # exactly as it would after a crash between "graph finished" and
            # "transcript committed". ---
            graph = client.app.state.graph
            final = run_turn(graph, thread_id, QUESTION)
            assert final.get("turn_id"), "graph did not stamp a turn_id"
            assert final.get("final_status") in {"accepted", "not_reviewed"}, final.get("final_status")

            pool = client.app.state.state_pool
            with pool.connection() as conn:
                before = conn.execute(
                    "SELECT count(*) AS n FROM rag_app.turns WHERE thread_id = %s", (thread_id,)
                ).fetchone()["n"]
                message_count_before = conn.execute(
                    "SELECT count(*) AS n FROM rag_app.messages WHERE thread_id = %s", (thread_id,)
                ).fetchone()["n"]
            assert before == 0, "turn should not be committed yet"
            assert message_count_before == 0, "transcript should not be written yet"

            recovered = client.get(f"/threads/{thread_id}/recover", headers=headers)
            recovered.raise_for_status()
            body = recovered.json()
            assert body["status"] == "recovered", body
            assert body["recovery"]["turn_id"] == final["turn_id"]
            assert body["recovery"]["already_committed"] is False

            with pool.connection() as conn:
                after = conn.execute(
                    "SELECT count(*) AS n FROM rag_app.turns WHERE thread_id = %s", (thread_id,)
                ).fetchone()["n"]
                messages_after = conn.execute(
                    "SELECT count(*) AS n FROM rag_app.messages WHERE thread_id = %s", (thread_id,)
                ).fetchone()["n"]
            assert after == 1, f"expected exactly one committed turn, got {after}"
            assert messages_after == 2, f"expected question+answer, got {messages_after}"

            # --- kill-point 4: recover a thread that is already committed --
            # must be a no-op, not a duplicate. ---
            again = client.get(f"/threads/{thread_id}/recover", headers=headers)
            again.raise_for_status()
            again_body = again.json()
            assert again_body["status"] == "recovered"
            assert again_body["recovery"]["already_committed"] is True

            with pool.connection() as conn:
                still = conn.execute(
                    "SELECT count(*) AS n FROM rag_app.turns WHERE thread_id = %s", (thread_id,)
                ).fetchone()["n"]
                messages_still = conn.execute(
                    "SELECT count(*) AS n FROM rag_app.messages WHERE thread_id = %s", (thread_id,)
                ).fetchone()["n"]
            assert still == 1, f"recover-after-commit duplicated the turn row: {still}"
            assert messages_still == 2, f"recover-after-commit duplicated messages: {messages_still}"

        print("Block C recovery verification passed")
        print(f"thread_id={thread_id}")
        print(f"turn_id={final['turn_id']}")
        print("mid_node_death_reports_unavailable=passed")
        print("kill_point_3_uncommitted_terminal_checkpoint=passed")
        print("kill_point_4_idempotent_recover_after_commit=passed")
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


if __name__ == "__main__":
    main()
