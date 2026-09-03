"""LLM-as-judge: does an answer assert a given claim?

Judged one claim at a time. Narrow questions are more reliable than asking a
model to score a whole answer at once, and a per-claim verdict is something a
human can check in seconds — which matters, because an unvalidated judge silently
corrupts every number downstream.

The judge runs on a different model *family* than the generator (Qwen vs
GPT-OSS), not merely a different size: a model grading its own output has a
documented self-preference bias, and same-family models share it in part.

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

JUDGE_MODEL = "qwen/qwen3.8-27b"
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
    import groq

    if not os.environ.get("GROQ_API_KEY"):
        raise MissingCredentials("GROQ_API_KEY is not set")
    return groq.Groq()


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


DECOMPOSE_PROMPT = """Break this answer into atomic factual claims — one \
verifiable statement each, no compound sentences. Ignore hedges, restatements of \
the question, and structural text.

ANSWER:
{answer}

Reply with only a JSON array of strings, no other text."""

# One call per answer, not per claim. Sending the full context alongside every
# individual claim multiplied token spend by the number of claims — ~15x on a
# typical answer — and exhausted a daily quota in five cases.
SUPPORT_BATCH_PROMPT = """Decide which CLAIMS are supported by the CONTEXT.

CONTEXT:
{context}

CLAIMS:
{claims}

For each numbered claim, decide whether it is stated in, or directly entailed \
by, the CONTEXT. Plausibility is not support — if the CONTEXT does not establish \
it, answer "unsupported", even if the claim sounds correct.

Reply with only a JSON array, one object per claim, in the same order:
[{{"n": 1, "verdict": "supported" or "unsupported"}}, ...]"""

RELEVANCE_PROMPT = """Does the ANSWER address the QUESTION that was asked?

QUESTION:
{question}

ANSWER:
{answer}

Judge only whether it responds to what was asked — not whether it is correct. \
An answer that is on-topic but wrong still addresses the question. An answer \
that discusses something else, or only restates the question, does not.

Reply with only a JSON object:
{{"verdict": "addresses" or "does_not_address", "evidence": ""}}"""


def _ask(prompt: str, model: str, cache_dir: str, tag: str, max_tokens: int = 1500):
    """Run one cached judge call, returning the parsed JSON object."""
    cache = Path(cache_dir) / f"{tag}_{hashlib.sha256((model + prompt).encode()).hexdigest()[:20]}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    response = _client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content or ""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    data = json.loads(cleaned)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def decompose_claims(answer: str, model: str = JUDGE_MODEL,
                     cache_dir: str = CACHE_DIR) -> list[str]:
    """Split an answer into atomic claims for groundedness checking."""
    data = _ask(DECOMPOSE_PROMPT.format(answer=answer), model, cache_dir, "decomp", 3000)
    return [str(c) for c in data] if isinstance(data, list) else []


def check_supported_batch(claims: list[str], context: str,
                          model: str = JUDGE_MODEL,
                          cache_dir: str = CACHE_DIR) -> list[bool]:
    """Which of these claims does the retrieved context establish?

    The reverse question from `judge_claim`, and it catches what forbidden_claims
    structurally cannot: an invention nobody thought to forbid.

    A short reply is missing verdicts rather than wrong ones, so unanswered
    claims default to unsupported — the conservative direction, since counting an
    unjudged claim as grounded would understate the problem.
    """
    if not claims:
        return []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    data = _ask(SUPPORT_BATCH_PROMPT.format(context=context, claims=numbered),
                model, cache_dir, "support", max_tokens=2000)

    verdicts = {int(r["n"]): r.get("verdict") == "supported"
                for r in data if isinstance(r, dict) and "n" in r}
    return [verdicts.get(i, False) for i in range(1, len(claims) + 1)]


def check_relevance(question: str, answer: str, model: str = JUDGE_MODEL,
                    cache_dir: str = CACHE_DIR) -> bool:
    data = _ask(RELEVANCE_PROMPT.format(question=question, answer=answer),
                model, cache_dir, "relev")
    return data.get("verdict") == "addresses"


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

    response = _client().chat.completions.create(
        model=model,
        max_tokens=1000,
        temperature=0,  # deterministic verdicts; a judge that wanders is unusable
        messages=[{
            "role": "user",
            "content": PROMPT.format(claim=claim, answer=answer),
        }],
    )
    raw = response.choices[0].message.content or ""
    data = _parse(raw)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data["verdict"] == "present", data.get("evidence", "")
