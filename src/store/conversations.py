"""SQLite log of every question asked through the chat, for later scoring.

Two kinds of user input reach the graph, and only one of them is a question:

* A **clarification** answers the resolver's own prompt ("which lender?"). It
  carries no information need of its own, so it is folded into the question it
  clarifies — one row, whose `resolved_question` is the complete question.
* A **follow-up** is a new information need on the same thread. It gets its
  own row, whether or not ground truth exists for it.

Scoring runs offline rather than inside a chat request: the judge costs Groq
tokens against a daily cap, it would add latency to every turn, and keeping it
separate means the whole log can be re-scored after ground truth grows.

This SQLite table is the offline scoring/audit log. The live LangGraph state
and user-visible transcript are independently durable in PostgreSQL; keeping
this secondary log preserves the earlier eval workflow without making SQLite
the conversation-memory authority.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.ingest.manifest import PROJECT_ROOT, load_manifest

DB_PATH = "data/conversations.db"
DEFAULT_MANIFEST = "config/corpus_manifest.json"
GROUND_TRUTH_FILES = ("eval/ground_truth_v2.json",)

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id              TEXT PRIMARY KEY,
    thread_id            TEXT NOT NULL,
    turn_index           INTEGER NOT NULL,
    kind                 TEXT NOT NULL CHECK (kind IN ('initial', 'followup')),
    raw_question         TEXT NOT NULL,
    resolved_question    TEXT NOT NULL,
    clarifications       TEXT NOT NULL DEFAULT '[]',
    clarification_rounds INTEGER NOT NULL DEFAULT 0,
    scope_status         TEXT,
    lender_id            TEXT,
    document_version     TEXT,
    source_doc           TEXT,
    answer               TEXT,
    citations            TEXT NOT NULL DEFAULT '[]',
    retrieved_ids        TEXT NOT NULL DEFAULT '[]',
    sub_queries          TEXT NOT NULL DEFAULT '[]',
    comparison_scopes    TEXT NOT NULL DEFAULT '[]',
    retrieved_by_scope   TEXT NOT NULL DEFAULT '{}',
    latency_ms           INTEGER,
    langsmith_run_id     TEXT,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_thread ON turns (thread_id, turn_index);

-- Ground truth itself stays in eval/*.json and is never copied here: it is the
-- user's file, it is still growing, and a copy would drift from it.
CREATE TABLE IF NOT EXISTS gt_links (
    turn_id          TEXT PRIMARY KEY REFERENCES turns (turn_id),
    case_id          TEXT NOT NULL,
    match_method     TEXT NOT NULL,
    -- 0 when the turn resolved to a different manual than the one the case's
    -- evidence lives in. Such a turn must not be scored: the system answering
    -- correctly from the current version would otherwise score zero against
    -- ground truth written for the historical one.
    scope_comparable INTEGER NOT NULL,
    scope_note       TEXT,
    linked_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    turn_id           TEXT PRIMARY KEY REFERENCES turns (turn_id),
    case_id           TEXT NOT NULL,
    claims_required   INTEGER,
    claims_present    INTEGER,
    claim_recall      REAL,
    hallucinated      INTEGER,
    citation_validity REAL,
    groundedness      REAL,
    metrics_json      TEXT,
    judge_model       TEXT,
    scored_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS guardrail_checks (
    request_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL,
    events_json TEXT NOT NULL,
    usage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    db = Path(path)
    if db.parent != Path("."):
        db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if "usage_json" not in {row[1] for row in conn.execute("PRAGMA table_info(guardrail_checks)")}:
        conn.execute("ALTER TABLE guardrail_checks ADD COLUMN usage_json TEXT NOT NULL DEFAULT '{}'")
    turn_columns = {row[1] for row in conn.execute("PRAGMA table_info(turns)")}
    if "comparison_scopes" not in turn_columns:
        conn.execute("ALTER TABLE turns ADD COLUMN comparison_scopes TEXT NOT NULL DEFAULT '[]'")
    if "retrieved_by_scope" not in turn_columns:
        conn.execute("ALTER TABLE turns ADD COLUMN retrieved_by_scope TEXT NOT NULL DEFAULT '{}'")
    return conn


# --------------------------------------------------------------------------
# ground truth: read-only, indexed by the question as typed
# --------------------------------------------------------------------------

def normalize(question: str) -> str:
    """Collapse a question to a matchable key.

    Matching is on the question as *typed*, never the resolved form: a resolved
    question carries an appended "NovaCred V0" that no ground-truth entry has,
    so it would never match.
    """
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


@lru_cache(maxsize=1)
def _manifest_by_scope() -> dict[tuple[str, str], str]:
    specs = load_manifest(DEFAULT_MANIFEST, require_files=False)
    return {(s.lender_id, s.document_version): s.source_doc for s in specs}


@lru_cache(maxsize=1)
def ground_truth() -> dict[str, dict]:
    """Load every ground-truth case, keyed by normalized question.

    Cached, so growing the eval set needs a process restart (or a cache_clear)
    to be picked up — deliberate, so a long chat session scores against one
    fixed snapshot rather than a set that shifts underneath it.
    """
    index: dict[str, dict] = {}
    for name in GROUND_TRUTH_FILES:
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        for case in json.loads(path.read_text())["cases"]:
            docs = {
                cid.split("::")[0]
                for group in case.get("evidence_groups", [])
                for cid in group.get("any_of_chunk_ids", [])
            }
            index[normalize(case["question"])] = {
                "case_id": case["id"],
                "evidence_docs": sorted(docs),
            }
    return index


def match_case(raw_question: str) -> dict | None:
    return ground_truth().get(normalize(raw_question))


def scope_comparability(source_doc: str | None, evidence_docs: list[str]):
    """Can a turn answered from `source_doc` be scored against this case?

    Only when the manual the turn actually searched is the one the case's
    evidence lives in. Anything else is not a failure to record as zero — it is
    a comparison that cannot be made.
    """
    if not evidence_docs:
        return False, "case has no evidence chunks"
    if source_doc is None:
        return False, "turn resolved no scope"
    if set(evidence_docs) == {source_doc}:
        return True, None
    return False, f"turn searched {source_doc}, evidence is in {', '.join(evidence_docs)}"


# --------------------------------------------------------------------------
# writing turns
# --------------------------------------------------------------------------

def next_turn_index(conn: sqlite3.Connection, thread_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 AS n FROM turns WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    return int(row["n"])


def record_turn(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    raw_question: str,
    state: dict,
    latency_ms: int | None = None,
    langsmith_run_id: str | None = None,
) -> str:
    """Store one completed turn and link it to ground truth when it matches.

    `state` is the graph's final state, so everything recorded here is what the
    pipeline actually did rather than a reconstruction of it.
    """
    from src.guardrails.privacy import sanitize, redact
    state = sanitize(state, "conversation_write")
    raw_question = redact(raw_question, "conversation_write")
    turn_id = str(uuid.uuid4())
    index = next_turn_index(conn, thread_id)
    scope = state.get("scope") or {}
    lender = scope.get("lender_id")
    version = scope.get("document_version")
    source_doc = _manifest_by_scope().get((lender, version)) if lender and version else None

    conn.execute(
        """INSERT INTO turns (
               turn_id, thread_id, turn_index, kind, raw_question,
               resolved_question, clarifications, clarification_rounds,
               scope_status, lender_id, document_version, source_doc,
               answer, citations, retrieved_ids, sub_queries,
               comparison_scopes, retrieved_by_scope,
               latency_ms, langsmith_run_id, created_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            turn_id, thread_id, index,
            "initial" if index == 0 else "followup",
            raw_question,
            state.get("resolved_question") or raw_question,
            json.dumps(state.get("clarifications", [])),
            state.get("clarification_rounds", 0),
            scope.get("status"), lender, version, source_doc,
            state.get("answer"),
            json.dumps(_citations(state.get("answer") or "")),
            json.dumps([c["chunk_id"] for c, _ in state.get("reranked", [])]),
            json.dumps(state.get("sub_queries", [])),
            json.dumps(state.get("comparison_scopes", [])),
            json.dumps({
                key: [chunk["chunk_id"] for chunk in chunks]
                for key, chunks in (state.get("comparison_context") or {}).items()
            }),
            latency_ms, langsmith_run_id, _now(),
        ),
    )

    matched = match_case(raw_question)
    if matched:
        comparable, note = scope_comparability(source_doc, matched["evidence_docs"])
        conn.execute(
            """INSERT INTO gt_links
               (turn_id, case_id, match_method, scope_comparable, scope_note, linked_at)
               VALUES (?,?,?,?,?,?)""",
            (turn_id, matched["case_id"], "exact_normalized",
             int(comparable), note, _now()),
        )
    conn.commit()
    return turn_id


def record_guards(conn: sqlite3.Connection, thread_id: str, state: dict) -> str:
    """Record blocked/clarification/completed requests without their content."""
    from src.guardrails.core import GuardEvent
    events = [GuardEvent.model_validate(e).model_dump() for e in state.get("guard_events", [])]
    request_id = str(uuid.uuid4())
    conn.execute("INSERT INTO guardrail_checks (request_id, thread_id, status, events_json, usage_json, created_at) VALUES (?,?,?,?,?,?)",
                 (request_id, thread_id, state.get("final_status", "clarification"),
                  json.dumps(events), json.dumps(state.get("request_usage", {})), _now()))
    conn.commit()
    return request_id


CITATION_RE = re.compile(r"[\[【]([^\]】\s]+::\d+)[\]】]")


def _citations(answer: str) -> list[str]:
    return sorted(set(CITATION_RE.findall(answer)))


# --------------------------------------------------------------------------
# reading back
# --------------------------------------------------------------------------

def scorable_turns(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Turns with matched, scope-comparable ground truth that are not yet scored."""
    return conn.execute(
        """SELECT t.*, g.case_id
             FROM turns t
             JOIN gt_links g ON g.turn_id = t.turn_id
        LEFT JOIN scores s ON s.turn_id = t.turn_id
            WHERE g.scope_comparable = 1
              AND s.turn_id IS NULL
              AND t.answer IS NOT NULL
         ORDER BY t.created_at"""
    ).fetchall()


def coverage(conn: sqlite3.Connection) -> dict:
    """How much of the log is scorable, and why the rest is not."""
    total = conn.execute("SELECT COUNT(*) AS n FROM turns").fetchone()["n"]
    linked = conn.execute("SELECT COUNT(*) AS n FROM gt_links").fetchone()["n"]
    comparable = conn.execute(
        "SELECT COUNT(*) AS n FROM gt_links WHERE scope_comparable = 1"
    ).fetchone()["n"]
    scored = conn.execute("SELECT COUNT(*) AS n FROM scores").fetchone()["n"]
    reasons = defaultdict(int)
    for row in conn.execute(
        "SELECT scope_note FROM gt_links WHERE scope_comparable = 0"
    ):
        reasons[row["scope_note"] or "unknown"] += 1
    return {
        "turns": total,
        "matched_ground_truth": linked,
        "unmatched": total - linked,
        "scope_comparable": comparable,
        "not_comparable": dict(reasons),
        "scored": scored,
        "awaiting_scoring": comparable - scored,
    }
