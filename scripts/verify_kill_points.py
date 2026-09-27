#!/usr/bin/env python3
"""Block C kill-points 1 and 2: a REAL `kill -9` on a REAL server process.

scripts/verify_recovery.py covers kill-points 3 and 4 with FastAPI's
TestClient, which runs the whole graph in-process -- it can never actually die
mid-node, only simulate what a terminal-but-uncommitted checkpoint looks like.

This script instead starts `uvicorn` as its own OS process, fires a real HTTP
request at it, and SIGKILLs that process while it is genuinely frozen inside
`retrieve_node` or `generate_node` (src/graph/pipeline.py) -- the two kill
points a TestClient cannot simulate truthfully. The freeze itself is real
code (`_kill_test_pause`, gated behind RAG_KILL_TEST_PAUSE, a no-op for every
real user and every other test) so the kill lands deterministically inside
the node under test instead of racing real network latency.

Uses the live NovaCred/LumenTrail Qdrant collection and real Groq/Voyage
calls -- this is not mocked.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = 8991
BASE = f"http://127.0.0.1:{PORT}"
QUESTION = "For NovaCred V1 explain the Blocked SSN rule."


def _start_server(pause_node: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["RAG_KILL_TEST_PAUSE"] = pause_node
    env["LANGCHAIN_TRACING_V2"] = "false"
    env["LANGSMITH_TRACING"] = "false"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.app.server:app", "--port", str(PORT)],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for _ in range(60):
        if proc.poll() is not None:
            out = proc.stdout.read()
            raise RuntimeError(f"server process died on startup:\n{out}")
        try:
            if httpx.get(f"{BASE}/openapi.json", timeout=1).status_code < 500:
                return proc
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not come up in time")


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)


def run_kill_point(node: str) -> dict:
    print(f"\n=== kill-point: mid-{node} ===")
    proc = _start_server(node)
    headers = {"X-Demo-User": f"kill-{node}-{uuid.uuid4().hex[:8]}"}
    try:
        created = httpx.post(f"{BASE}/threads", headers=headers, timeout=10)
        created.raise_for_status()
        thread_id = created.json()["thread_id"]
        print(f"thread_id={thread_id}")

        result_box: dict = {}

        def fire() -> None:
            try:
                result_box["resp"] = httpx.post(
                    f"{BASE}/chat", headers=headers, timeout=60,
                    json={"thread_id": thread_id, "message": QUESTION},
                )
            except httpx.TransportError as exc:
                result_box["killed_mid_request"] = type(exc).__name__

        t = threading.Thread(target=fire, daemon=True)
        t.start()
        time.sleep(4)  # node hook sleeps 30s -- 4s in, we're genuinely inside it
        print(f"killing pid={proc.pid} with SIGKILL while it should be inside {node}_node")
        _kill(proc)
        t.join(timeout=5)
        print(f"in-flight request outcome: {result_box}")
    finally:
        _kill(proc)

    # Fresh process, no pause -- ask what a truly-dead-mid-node thread recovers to.
    proc2 = _start_server("")
    try:
        recovered = httpx.get(f"{BASE}/threads/{thread_id}/recover", headers=headers, timeout=10)
        recovered.raise_for_status()
        body = recovered.json()
        print(f"GET /recover after real kill -> {json.dumps(body, indent=2)}")
        assert body["status"] == "recovery_unavailable", (
            f"expected recovery_unavailable, got {body}"
        )
        print(f"mid_{node}_kill: correctly reports recovery_unavailable "
              f"(no terminal checkpoint existed to recover)")
        return body
    finally:
        _kill(proc2)


def main() -> None:
    run_kill_point("retrieve")
    run_kill_point("generate")
    print("\nBoth real kill-points verified against a truly killed -9 process:")
    print("LangGraph only checkpoints AFTER a node returns, so a death inside")
    print("retrieve_node or generate_node leaves the checkpoint at the end of")
    print("whatever node finished last. There is no partial in-node state to")
    print("recover, and /recover correctly refuses rather than fabricating one.")


if __name__ == "__main__":
    main()
