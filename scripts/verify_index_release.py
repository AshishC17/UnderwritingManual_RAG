"""Rehearse alias promotion/rollback and checkpoint pinning, with no LLM calls.

This deliberately exercises graph/checkpointer mechanics, not /chat or answer
quality. Original uw_manual is never written. All test checkpoints are retained
under a new index-rehearsal-* thread for inspection. Always attempt rollback.
"""
import argparse
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from src.graph.pipeline import build_graph
from src.store import index_builds as ib, qdrant_store as qs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--alias", default=ib.ACTIVE_ALIAS)
    p.add_argument("--postgres", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("output already exists")
    load_dotenv(ROOT / ".env")
    with ExitStack() as stack:
        client = stack.enter_context(closing(qs.connect()))
        before = ib.resolve_active(client, a.alias)
        saver = (stack.enter_context(PostgresSaver.from_conn_string(os.environ.get(
            "STATE_DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5432/rag"))) if a.postgres else MemorySaver())
        selector = lambda: ib.resolve_active(client, a.alias)
        graph = build_graph(checkpointer=saver, index_selector=selector)
        thread_id = "index-rehearsal-" + uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        first = graph.invoke({"question": "What does Code 128 mean?", "history": []}, config)
        assert first.get("__interrupt__"), "expected deterministic missing-lender clarification"
        assert first["index_binding"] == before
        promoted = False
        try:
            release = ib.release(client, a.candidate, alias=a.alias,
                                 expected_collection=before["collection"], evaluation=a.evaluation,
                                 reason="Identical-index lifecycle rehearsal")
            promoted = True
            # Reconstruct the graph. With --postgres, also close/reopen the
            # saver connection to demonstrate reading the durable checkpoint.
            if a.postgres:
                saver2 = stack.enter_context(PostgresSaver.from_conn_string(os.environ.get(
                    "STATE_DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5432/rag")))
            else:
                saver2 = saver
            restarted = build_graph(checkpointer=saver2, index_selector=selector)
            loaded = restarted.get_state(config).values
            assert loaded["index_binding"] == before
            resumed = restarted.invoke(Command(resume="I am not sure"), config)
            assert resumed["index_binding"] == before
            finished = restarted.invoke(Command(resume="I am still not sure"), config)
            assert not finished.get("__interrupt__")
            assert finished["index_binding"] == before
            fresh = restarted.invoke({"question": "Which lenders are in the corpus?", "history": []}, config)
            assert fresh["index_binding"]["build_id"] == a.candidate
            assert fresh["behavior"] == "list_corpus_lenders"
            report = {"thread_id": thread_id, "checkpoint_backend": "postgres" if a.postgres else "memory",
                      "provider_calls": 0, "initial_build": before["build_id"],
                      "after_recompile_build": loaded["index_binding"]["build_id"],
                      "resumed_build": resumed["index_binding"]["build_id"],
                      "new_turn_build": fresh["index_binding"]["build_id"],
                      "promotion": release, "qualification": "Graph/checkpoint lifecycle test; no HTTP or generated-answer quality test."}
        finally:
            if promoted:
                rollback = ib.release(client, before["build_id"], alias=a.alias,
                                      expected_collection=ib.binding_for(a.candidate)["collection"],
                                      rollback=True, reason="Return learning alias to baseline after rehearsal")
        report["rollback"] = rollback
        report["final_active_build"] = ib.resolve_active(client, a.alias)["build_id"]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        ib.write_once(a.out, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
