"""Proactive pacing for Voyage calls.

Voyage's free tier allows 3 requests/min and 10K tokens/min. Both embedding and
reranking draw on it — assumed to be one shared bucket rather than per-endpoint,
which has NOT been verified against Voyage's docs; if it turns out to be
per-endpoint this pacing is merely conservative, not wrong.

Proactive pacing rather than reactive backoff: this project established
repeatedly that a rejected request still consumes the quota it was denied for,
so retrying into an exhausted window makes the exhaustion worse. Waiting *before*
the call costs the same wall-clock time and never burns budget on a failure.

Deliberately not a token-bucket: request count is the binding constraint here (6
requests needs 2 min at 3/min, while ~9.2K tokens needs ~1 min at 10K/min), and
a simple minimum-interval gate is easier to reason about than two interacting
budgets. If token spend ever becomes binding, that assumption needs revisiting.
"""

from __future__ import annotations

import threading
import time
from src.util.telemetry import wait_event

# 3 requests/min = one per 20s. The extra second absorbs clock skew between the
# local timer and Voyage's own window accounting.
MIN_INTERVAL_S = 21.0

_lock = threading.Lock()
_last_call_at = 0.0


def voyage_gate(verbose: bool = False) -> float:
    """Block until it is safe to make another Voyage request.

    Returns the seconds actually waited, so callers can report pacing without
    timing it themselves. Call this immediately before a real API request —
    never before a cache lookup, or cached runs would inherit the rate limit of
    uncached ones and a fully-cached eval would crawl for no reason.
    """
    global _last_call_at
    with _lock:
        now = time.monotonic()
        wait = MIN_INTERVAL_S - (now - _last_call_at)
        if wait <= 0:
            _last_call_at = now
            return 0.0
        if verbose:
            print(f"      pacing Voyage: waiting {wait:.0f}s", flush=True)
        time.sleep(wait)
        wait_event("voyage", wait, "proactive_pacing")
        _last_call_at = time.monotonic()
        return wait


def reset() -> None:
    """Forget the last-call timestamp. For tests only."""
    global _last_call_at
    with _lock:
        _last_call_at = 0.0
