"""Per-node observations. No global request data; prices are optional inputs.

API calls mean SDK invocations, not unobservable internal HTTP retries. Cost is
an API estimate, not an invoice and not Qdrant/hosting infrastructure cost.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

_events: ContextVar[list | None] = ContextVar("rag_usage_events", default=None)
_request_events: ContextVar[list | None] = ContextVar("rag_request_usage", default=None)
PRICES = Path(__file__).resolve().parents[2] / "config/usage_prices.json"


def record(event: dict) -> None:
    events = _events.get()
    if events is not None:
        events.append(event)
    request_events = _request_events.get()
    if request_events is not None:
        request_events.append(event)


@contextmanager
def capture_request():
    """Include calls in failed nodes and guards outside the graph, once each."""
    events = []
    token = _request_events.set(events)
    try:
        yield events
    finally:
        _request_events.reset(token)


@contextmanager
def capture():
    events = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


def cache_hit(provider: str, operation: str, model: str) -> None:
    record({"kind": "cache_hit", "provider": provider, "operation": operation, "model": model})


def wait_event(provider: str, seconds: float, reason: str) -> None:
    record({"kind": "wait", "provider": provider, "elapsed_ms": round(seconds * 1000, 2), "reason": reason})


def tracked_call(provider: str, operation: str, model_name: str, function, *args, **kwargs):
    from src.guardrails.privacy import sanitize, redact
    args = sanitize(args, "provider_input")
    kwargs = sanitize(kwargs, "provider_input")
    reservation = None
    if provider == "groq" and model_name.startswith("qwen/"):
        from src.util.groq_ratelimit import reserve, reconcile
        # All Qwen consumers share this window: guards, decomposition,
        # follow-up classification and answer review.
        prompt = json.dumps(kwargs.get("messages", []), ensure_ascii=False)
        reservation = reserve(model_name, prompt, kwargs.get("max_tokens", 300),
                              max_wait_seconds=65 if operation == "guardrail_screen" else None)
    started = time.perf_counter()
    event = {"kind": "api_call", "provider": provider, "operation": operation, "model": model_name,
             "input_tokens": None, "output_tokens": None, "total_tokens": None,
             "status": "error", "internal_http_retries": None}
    try:
        result = function(*args, **kwargs)
        # Clean model-generated text before callers parse/cache it. Embedding
        # and rerank numeric responses are left intact.
        for choice in getattr(result, "choices", []) or []:
            message = getattr(choice, "message", None)
            for field in ("content", "reasoning"):
                content = getattr(message, field, None)
                if isinstance(content, str):
                    setattr(message, field, redact(content, "provider_output"))
        usage = getattr(result, "usage", None)
        if usage is not None:
            for field, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens")):
                value = getattr(usage, source, None)
                event[field] = value if type(value) is int and value >= 0 else None
        elif type(getattr(result, "total_tokens", None)) is int and result.total_tokens >= 0:
            event["total_tokens"] = result.total_tokens
        event["status"] = "ok"
        if reservation is not None:
            reconcile(model_name, reservation, event["total_tokens"])
        return result
    except Exception as exc:
        event["error_type"] = type(exc).__name__  # never log credentials/response bodies
        details = rate_limit_details(exc)
        if details:
            event["rate_limit"] = details
        raise
    finally:
        event["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record(event)


def rate_limit_details(exc: Exception) -> dict | None:
    """Return allowlisted rate-limit diagnostics without raw bodies or URLs."""
    if type(exc).__name__ != "RateLimitError":
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    allow = (
        "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens", "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
        "retry-after", "retry-after-ms",
    )
    safe_headers = {name: str(headers[name]) for name in allow if name in headers}
    body = getattr(exc, "body", None)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    text = " ".join(str(x) for x in (
        error.get("message", ""), error.get("type", ""), error.get("code", "")
    )).lower()
    if "tokens per minute" in text or "tpm" in text:
        category = "tokens_per_minute"
    elif "tokens per day" in text or "tpd" in text:
        category = "tokens_per_day"
    elif "requests per minute" in text or "rpm" in text:
        category = "requests_per_minute"
    elif "requests per day" in text or "rpd" in text:
        category = "requests_per_day"
    elif "too large" in text or "maximum context" in text:
        category = "request_too_large"
    else:
        category = "unknown_rate_limit"
    result = {"category": category, "headers": safe_headers}
    # Groq's rate message can distinguish a too-large individual request from
    # an exhausted window. Persist only labelled numeric values, never the raw
    # provider message (which can contain request data on other error types).
    for label, key in (("limit", "reported_limit"), ("requested", "reported_requested")):
        match = re.search(rf"\b{label}\s*[:=]?\s*([\d,]+)", text)
        if match:
            result[key] = int(match.group(1).replace(",", ""))
    if error.get("type") is not None:
        result["provider_error_type"] = str(error["type"])[:80]
    if error.get("code") is not None:
        result["provider_error_code"] = str(error["code"])[:80]
    return result


def local_tool(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record({"kind": "local_tool", "operation": name,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
        return wrapped
    return decorate


def observed_node(name: str, function):
    @wraps(function)
    def wrapped(state):
        started = time.perf_counter()
        wall = datetime.now(timezone.utc).isoformat()
        with capture() as events:
            result = function(state)
        event = {"node": name, "started_at": wall,
                 "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "events": events}
        return {**result, "node_events": [*result.get("node_events", state.get("node_events", [])), event]}
    return wrapped


def load_prices() -> dict:
    try:
        return json.loads(PRICES.read_text())
    except (OSError, ValueError):
        return {"as_of": None, "models": {}}


def estimate_cost(call: dict, prices: dict) -> float | None:
    rate = prices.get("models", {}).get(f"{call['provider']}:{call['model']}")
    if not rate or call.get("status") != "ok":
        return None
    if call.get("input_tokens") is not None and call.get("output_tokens") is not None:
        fields = [("input_per_million", call["input_tokens"]), ("output_per_million", call["output_tokens"])]
    elif call.get("total_tokens") is not None:
        fields = [("total_per_million", call["total_tokens"])]
    else:
        return None
    if any(type(rate.get(key)) not in (int, float) or rate[key] < 0 for key, _ in fields):
        return None
    return sum(rate[key] * count / 1_000_000 for key, count in fields)


def summarize(nodes: list[dict], prices: dict | None = None) -> dict:
    prices = load_prices() if prices is None else prices
    events = [e for n in nodes for e in n["events"]]
    calls = [e for e in events if e["kind"] == "api_call"]
    costs = [estimate_cost(e, prices) for e in calls]
    known_cost = sum(c for c in costs if c is not None)
    missing_tokens = sum(e["total_tokens"] is None for e in calls)
    return {
        "node_elapsed_ms": round(sum(n["elapsed_ms"] for n in nodes), 2),
        "api_calls": len(calls), "cache_hits": sum(e["kind"] == "cache_hit" for e in events),
        "local_tool_calls": sum(e["kind"] == "local_tool" for e in events),
        "wait_ms": round(sum(e["elapsed_ms"] for e in events if e["kind"] == "wait"), 2),
        "known_total_tokens": sum(e["total_tokens"] or 0 for e in calls),
        "calls_without_token_usage": missing_tokens,
        "estimated_api_cost_usd": None if any(c is None for c in costs) else known_cost,
        "known_api_cost_subtotal_usd": known_cost,
        "calls_without_cost_estimate": sum(c is None for c in costs),
        "price_as_of": prices.get("as_of"), "internal_http_retries": None,
        "measurement_scope": "completed graph nodes; excludes human wait, interrupted/uncaught failed nodes and hosting costs",
    }
