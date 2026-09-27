"""Process-local Qwen token pacing for the observed Groq account limit.

The account reported 8,000 tokens/minute for the Qwen model on 2026-09-15.
This is configurable because limits vary by account/model. Reservations use a
conservative prompt estimate plus the requested output ceiling, then reconcile to
provider-reported actual usage after success. Multiple server processes still
need a shared external limiter.
"""
from __future__ import annotations

import math
import os
import threading
import time
import uuid

from src.util.telemetry import wait_event

TPM = int(os.environ.get("GROQ_QWEN_TPM", "8000"))
SAFETY_TOKENS = int(os.environ.get("GROQ_QWEN_TPM_SAFETY", "500"))
WINDOW_S = 60.0

_lock = threading.Lock()
_reservations: dict[str, list[dict]] = {}


def estimate_prompt_tokens(prompt: str) -> int:
    """Conservative English/JSON estimate; final accounting uses API usage."""
    return max(1, math.ceil(len(prompt) / 3.5))


def output_budget(prompt: str, wanted: int, minimum: int = 384) -> int:
    available = TPM - SAFETY_TOKENS - estimate_prompt_tokens(prompt)
    if available < minimum:
        raise ValueError(
            f"judge prompt estimate leaves fewer than {minimum} output tokens "
            f"under the configured {TPM} TPM limit"
        )
    return min(wanted, available)


def _prune(model: str, now: float) -> list[dict]:
    live = [r for r in _reservations.get(model, []) if now - r["at"] < WINDOW_S]
    _reservations[model] = live
    return live


class CapacityWaitTimeout(TimeoutError):
    """The local model quota could not be reserved within the wait budget."""


def reserve(model: str, prompt: str, max_output_tokens: int, *, max_wait_seconds: float | None = None) -> str:
    requested = estimate_prompt_tokens(prompt) + max_output_tokens
    capacity = TPM - SAFETY_TOKENS
    if requested > capacity:
        raise ValueError(f"estimated request {requested} exceeds safe Qwen TPM capacity {capacity}")
    started = time.monotonic()
    while True:
        with _lock:
            now = time.monotonic()
            live = _prune(model, now)
            if sum(r["tokens"] for r in live) + requested <= capacity:
                reservation_id = uuid.uuid4().hex
                live.append({"id": reservation_id, "at": now, "tokens": requested})
                return reservation_id
            wait = max(0.05, WINDOW_S - (now - min(r["at"] for r in live)) + 0.05)
            if max_wait_seconds is not None and now - started + wait > max_wait_seconds:
                raise CapacityWaitTimeout("Qwen quota wait budget exhausted")
        time.sleep(wait)
        wait_event("groq", wait, "qwen_tpm_pacing")


def reconcile(model: str, reservation_id: str, actual_total_tokens: int | None) -> None:
    if type(actual_total_tokens) is not int or actual_total_tokens < 0:
        return
    with _lock:
        for item in _reservations.get(model, []):
            if item["id"] == reservation_id:
                # Actual usage can exceed an estimate; retaining the smaller
                # number would undercount the occupied token window.
                item["tokens"] = actual_total_tokens
                break


def reset() -> None:
    """Testing only; production reservations expire naturally."""
    with _lock:
        _reservations.clear()
