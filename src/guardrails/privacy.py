"""Local Presidio recognizers. Deliberately excludes names, ages and dates.

PatternRecognizer requires no downloaded NLP model or remote service. These
recognizers cover selected identifiers, not every kind of personal data.
"""
from __future__ import annotations

import json
import hashlib
import time
from functools import lru_cache

from src.guardrails.core import GuardrailError, _privacy_cache, event, policy, reject


@lru_cache(maxsize=1)
def engines():
    from presidio_analyzer import Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    patterns = {
        "SSN": r"\b\d{3}[- ]\d{2}[- ]\d{4}\b|(?i:\bSSN\s*[:=#]?\s*(?:\d[ -]?){8}\d\b)",
        "EMAIL": r"(?i:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})",
        "PHONE": r"(?<![\w-])(?:\+1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}\b|(?i:\b(?:phone|mobile|tel)\s*[:=]?\s*\+?\d{10,15}\b)",
        "ACCOUNT": r"(?i:\b(?:account|acct|routing)(?:\s+(?:number|no\.?|id))?\s*[:=#]?\s*\d{6,18}\b)",
        "SECRET": r"\b(?:sk-|gsk_|ghp_)[A-Za-z0-9_-]{16,}\b|\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b|(?i:\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password)\s*[:=]\s*[\"']?[A-Za-z0-9_./+\-=]{8,})",
    }
    return [PatternRecognizer(supported_entity=kind, patterns=[Pattern(kind, regex, 0.8)])
            for kind, regex in patterns.items()], AnonymizerEngine()


def _redact(text: str) -> tuple[str, tuple[str, ...]]:
    from presidio_anonymizer.entities import OperatorConfig
    recognizers, anonymizer = engines()
    hits = [hit for r in recognizers for hit in r.analyze(text, [r.supported_entities[0]])]
    if not hits:
        return text, ()
    kinds = tuple(sorted({h.entity_type for h in hits}))
    result = anonymizer.anonymize(text=text, analyzer_results=hits,
                                 operators={k: OperatorConfig("replace", {"new_value": f"<{k}>"}) for k in kinds})
    return result.text, kinds


def redact(text: str, stage: str = "privacy") -> str:
    if len(text) > policy().max_field_chars:
        reject("field_too_large", stage)
    started = time.perf_counter()
    try:
        cache = _privacy_cache.get()
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = cache.get(key) if cache is not None else None
        clean, kinds = cached if cached is not None else _redact(text)
        if cache is not None and len(cache) < 2048:
            cache[key] = (clean, kinds)  # hash key, sanitised value, this request only
    except GuardrailError:
        raise
    except Exception:
        reject("privacy_unavailable", stage, error=True)
    if kinds:
        event("pii", stage, "redact", "personal_identifier", detected_types=list(kinds),
              latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return clean


def sanitize(value, stage: str = "privacy", depth: int = 0):
    if depth > 40:
        reject("payload_too_deep", stage)
    if isinstance(value, str):
        return redact(value, stage)
    if isinstance(value, dict):
        if "chunk_id" in value and "text" in value:
            source_text = json.dumps(value, ensure_ascii=False)
            if redact(source_text, stage) != source_text:
                reject("sensitive_source", stage)
        return {redact(k, stage) if isinstance(k, str) else k: "<SECRET>" if str(k).lower() in {"authorization", "api_key", "password", "access_token", "refresh_token", "cookie", "set-cookie"}
                else sanitize(v, stage, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [sanitize(v, stage, depth + 1) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def cache_json(value) -> str:
    return json.dumps(sanitize(value, "cache_write"), ensure_ascii=False)


def cache_read(text: str):
    return sanitize(json.loads(text), "cache_read")


def trace_payload(value):
    try:
        def serializable(item):
            if isinstance(item, dict):
                return {str(k): serializable(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [serializable(v) for v in item]
            if hasattr(item, "model_dump"):
                return serializable(item.model_dump(mode="json"))
            if item is None or isinstance(item, (str, int, float, bool)):
                return item
            return f"<{type(item).__name__}>"
        return sanitize(serializable(value), "trace")
    except Exception:
        return {"redacted": "privacy_check_unavailable"}
