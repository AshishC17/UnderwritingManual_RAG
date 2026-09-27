#!/usr/bin/env python3
"""Initialize LangGraph checkpoint and Block-A application tables once."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.store import state_store


DATABASE_URL = os.environ.get(
    "STATE_DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5432/rag"
)


def main() -> None:
    pool = ConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=2,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    pool.open(wait=True, timeout=20)
    try:
        PostgresSaver(pool).setup()
        state_store.migrate(pool)
        state_store.assert_ready(pool)
    finally:
        pool.close()
    print("PostgreSQL checkpoint and conversation schemas are ready.")


if __name__ == "__main__":
    main()
