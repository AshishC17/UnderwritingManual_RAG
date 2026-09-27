"""Narrow, source-aware routes for corpus and conversation questions.

These questions are not underwriting-policy lookups. A manual can identify its
own lender, but cannot prove which answer was most recent in a conversation.
Only high-confidence wording is routed here; other questions keep the normal
follow-up classifier and scoped RAG path.
"""

from __future__ import annotations

import re

from src.ingest.manifest import DocumentSpec, load_manifest

MANIFEST = "config/corpus_manifest.json"
POLICY_WORDS = re.compile(
    r"\b(?:rule|code|threshold|credit|freeze|prescreen|score|eligibility|"
    r"exception|income|funding|decision|disposition)\b", re.I
)
RECENT_WORDS = re.compile(r"\b(?:last|latest|most recent(?:ly)?|previous)\b", re.I)


def _legacy_conversation_question(question: str) -> bool:
    """Exclude old misrouted chat questions that have misleading chunk IDs."""
    text = question.lower()
    if not re.search(r"\b(?:answer|reply|response)\b", text):
        return False
    if re.search(r"\b(?:conversation|chat)\b", text):
        return True
    return bool(RECENT_WORDS.search(text)
                and re.search(r"\b(?:lender|lender-specific)\b", text)
                and not POLICY_WORDS.search(text))


def detect_meta_intent(question: str) -> str | None:
    """Recognize only explicit inventory or conversation-history requests.

    Deliberately avoid turning a cross-lender policy question into a catalog
    answer, or a question about a policy's latest version into a chat-history
    lookup. The ordinary classifier handles anything less explicit.
    """
    text = " ".join(question.lower().split())
    if POLICY_WORDS.search(text):
        return None
    if (re.search(r"\b(?:lenders|financial institutions)\b", text)
            and re.search(r"\b(?:corpus|manuals?|documents?|cover(?:ed|s)?|"
                          r"available|included|indexed|support(?:ed|s)?)\b", text)
            and re.match(r"^(?:which|what|list|name|show|tell me|give me|how many)\b", text)):
        return "list_corpus_lenders"

    if (RECENT_WORDS.search(text)
            and re.search(r"\b(?:lender|lender-specific)\b", text)
            and re.search(r"\b(?:answer|reply|response)\b", text)
            and re.match(r"^(?:which|what|who|tell me|remind me)\b", text)):
        # "What was your latest answer for NovaCred?" is a different request:
        # this route reports the latest lender-specific answer *overall*.
        specs = load_manifest(MANIFEST, require_files=False)
        if not any(re.search(rf"\b{re.escape(spec.lender_id)}\b", text)
                   or re.search(rf"\b{re.escape(spec.lender_name.lower())}\b", text)
                   for spec in specs):
            return "last_lender_answer"
    return None


def lender_inventory(specs: list[DocumentSpec] | None = None) -> str:
    specs = specs if specs is not None else load_manifest(MANIFEST, require_files=False)
    names = sorted({spec.lender_name for spec in specs})
    if len(names) == 1:
        return f"The underwriting corpus covers {names[0]}."
    return f"The underwriting corpus covers {', '.join(names[:-1])} and {names[-1]}."


def latest_lender_answer(history: list[dict], specs: list[DocumentSpec] | None = None) -> tuple[str, dict]:
    """Use a completed policy turn, never a retrieved manual, as provenance.

    New turns carry their resolved scope. Older PostgreSQL checkpoints contain
    only chunk IDs; map those IDs through the manifest without re-indexing or
    guessing from prose. Skip mixed or unknown source IDs rather than naming
    an unverified lender.
    """
    specs = specs if specs is not None else load_manifest(MANIFEST, require_files=False)
    by_source = {spec.source_doc: spec for spec in specs}
    for index in range(len(history) - 1, -1, -1):
        turn = history[index]
        if not turn.get("answer") or not turn.get("chunk_ids"):
            continue
        if turn.get("turn_type") not in (None, "policy_answer"):
            continue
        if turn.get("turn_type") is None and _legacy_conversation_question(turn.get("question", "")):
            continue

        scope = turn.get("scope") or {}
        if scope.get("status") == "resolved":
            lender, version = scope.get("lender_name"), scope.get("document_version")
        elif turn.get("turn_type") is None:  # Legacy checkpoint format.
            ids = turn["chunk_ids"]
            if not all(isinstance(cid, str) and "::" in cid for cid in ids):
                continue
            sources = {cid.rsplit("::", 1)[0] for cid in ids}
            if len(sources) != 1:
                continue
            spec = by_source.get(next(iter(sources)))
            if spec is None:
                continue
            lender, version = spec.lender_name, spec.document_version
        else:
            continue
        if not lender or not version:
            continue
        answer = f"The most recent lender-specific policy answer in this conversation was about {lender} {version}."
        if index != len(history) - 1:
            answer += " The immediately preceding reply was not a lender-specific policy answer."
        return answer, {"kind": "conversation_history", "turn_index": index}

    return ("I have not given a lender-specific policy answer in this conversation yet.",
            {"kind": "conversation_history", "turn_index": None})
