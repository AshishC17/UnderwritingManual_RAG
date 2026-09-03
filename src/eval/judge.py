"""LLM-as-judge: does an answer assert a given claim?

Judged one claim at a time. Narrow questions are more reliable than asking a
model to score a whole answer at once, and a per-claim verdict is something a
human can check in seconds — which matters, because an unvalidated judge silently
corrupts every number downstream.

The judge runs on a *different* model than the generator: a model grading its own
output has a documented self-preference bias.

The judge must quote the sentence it relies on. Forcing a span makes the verdict
checkable and measurably reduces judge hallucination — a model that has to point
at specific text invents less than one emitting a bare label.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

JUDGE_MODEL = "claude-opus-4-8"
CACHE_DIR = "data/interim/judgements"

PROMPT = """You are checking whether a specific claim is asserted in an answer.

CLAIM:
{claim}

ANSWER:
{answer}

Does the ANSWER assert this CLAIM, in substance? Paraphrase counts — identical \
wording is not required. Contradicting the claim is NOT asserting it. Mentioning \
the topic without asserting the claim is NOT asserting it.

Reply with only a JSON object, no other text:
{{"verdict": "present" or "absent", "evidence": "the exact sentence from ANSWER \
that asserts it, or empty string if absent"}}"""


class MissingCredentials(RuntimeError):
    pass


def _key(model: str, claim: str, answer: str) -> str:
    return hashlib.sha256(f"{model}|{claim}|{answer}".encode()).hexdigest()[:20]


def _client():
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise MissingCredentials("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic()


def _parse(text: str) -> dict:
    """Parse the judge's JSON, tolerating a code fence but nothing looser.

    A malformed reply raises rather than defaulting to "absent": silently
    scoring an unparseable judgement as absent would understate hallucination
    exactly when the judge is misbehaving.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(cleaned)
    if data.get("verdict") not in {"present", "absent"}:
        raise ValueError(f"unexpected verdict: {data.get('verdict')!r}")
    return data


def judge_claim(
    claim: str,
    answer: str,
    model: str = JUDGE_MODEL,
    cache_dir: str = CACHE_DIR,
) -> tuple[bool, str]:
    """Return (claim_is_present, supporting_sentence)."""
    cache = Path(cache_dir) / f"{_key(model, claim, answer)}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return data["verdict"] == "present", data.get("evidence", "")

    response = _client().messages.create(
        model=model,
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": PROMPT.format(claim=claim, answer=answer),
        }],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    data = _parse(raw)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data["verdict"] == "present", data.get("evidence", "")
