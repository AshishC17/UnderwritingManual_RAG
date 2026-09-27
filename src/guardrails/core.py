"""Versioned policy and value-free events. No request bodies in exceptions."""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: str
    max_input_chars: int = Field(gt=0)
    max_field_chars: int = Field(gt=0)
    max_context_chars: int = Field(gt=0)
    semantic_checks: bool
    semantic_model: str
    semantic_timeout_seconds: float = Field(gt=0, le=60)
    semantic_max_output_tokens: int = Field(gt=0, le=2000)
    semantic_cache_entries: int = Field(gt=0, le=10000)
    reviewed_image_sha256: list[str]


class GuardEvent(BaseModel):
    guard: str
    stage: str
    decision: Literal["allow", "redact", "block", "decline", "error", "disabled"]
    reason_code: str
    policy_version: str
    latency_ms: float = 0
    detected_types: list[str] = Field(default_factory=list)
    error: str | None = None
    rate_limit: dict | None = None


class GuardrailError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@lru_cache(maxsize=1)
def policy() -> Policy:
    path = Path(__file__).resolve().parents[2] / "config/guardrails.json"
    return Policy.model_validate(json.loads(path.read_text()))


def cache_directory(path: str | Path) -> str:
    """One guard-policy namespace, including calls through nested wrappers."""
    value = Path(path)
    return str(value if value.name == policy().policy_version else value / policy().policy_version)


_events: ContextVar[list | None] = ContextVar("guard_events", default=None)
_checked: ContextVar[set | None] = ContextVar("guard_checked", default=None)
_privacy_cache: ContextVar[dict | None] = ContextVar("guard_privacy_cache", default=None)


@contextmanager
def capture_guards():
    events = []
    t1, t2 = _events.set(events), _checked.set(set())
    t3 = _privacy_cache.set({})
    try:
        yield events
    finally:
        _privacy_cache.reset(t3)
        _checked.reset(t2)
        _events.reset(t1)


def event(guard: str, stage: str, decision: str, reason: str, **kwargs) -> None:
    events = _events.get()
    if events is not None:
        events.append(GuardEvent(guard=guard, stage=stage, decision=decision,
                                reason_code=reason, policy_version=policy().policy_version,
                                **kwargs).model_dump())


def reject(code: str, stage: str, *, error: bool = False) -> None:
    event("enforcement", stage, "error" if error else "block", code)
    raise GuardrailError(code)


def blocked_result(code: str) -> dict:
    if code == "out_of_domain":
        text = "This assistant answers questions about the NovaCred and LumenTrail underwriting manuals only."
    elif code == "semantic_guard_rate_limited":
        text = "The required safety check reached the model's usage limit. Please wait before trying again."
    elif code.endswith("unavailable"):
        text = "I couldn't complete the required checks. Please try again."
    elif code in {"wrong_scope", "sensitive_source", "invalid_citation", "unsafe_output", "empty_response", "evidence_item_too_large", "context_too_large"}:
        text = "I couldn't validate a response from the available evidence. No policy answer was released."
    elif code in {"empty_input", "input_too_large"}:
        text = "Please enter a nonempty question of at most 12,000 characters."
    else:
        text = "I couldn't safely process this request or its evidence. Please rephrase the request."
    return {"question": "", "resolved_question": "", "answer": text, "context_chunks": [],
            "reranked": [], "history": [], "sub_queries": [], "scope": {},
            "final_status": "scope_error" if code == "wrong_scope" else "guard_blocked",
            "guard_reason": code, "retry_attempt": 0, "recovery_exhausted": False,
            "usage_summary": {}, "node_events": []}
