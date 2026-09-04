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

JUDGE_MODEL = "qwen/qwen3.6-27b"
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


def _extract_json(text: str):
    """Pull the JSON payload out of a judge reply.

    Reasoning models (qwen3.x) emit a `<think>...</think>` block before the
    answer, so the reply does not begin with JSON. Strip that and any code
    fence, then take the first balanced object or array. A truncated reply has
    no closing tag and no parseable JSON — that raises rather than silently
    scoring as "absent", which would understate hallucination exactly when the
    judge is misbehaving.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.S)  # truncated block
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned).strip()

    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1),
                default=-1)
    if start == -1:
        raise ValueError(f"no JSON in judge reply: {text[:120]!r}")
    return json.loads(cleaned[start:])


def _parse(text: str) -> dict:
    data = _extract_json(text)
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


# Groq's real constraint for this model is `max_tokens <= 16384` — a hard per-
# response cap, unrelated to input size (context window is 131K). An earlier
# version of this file assumed input+output had to fit a shared ~9000 budget;
# that was wrong, conflated with a separate quota error, and it starved the
# model's reasoning on multi-claim batches, producing truncated <think> blocks
# with no JSON. qwen3.x spends real output tokens reasoning before it answers,
# so the cap needs headroom, not a tight fit.
MODEL_MAX_TOKENS = 16384
OUTPUT_SAFETY_MARGIN = 1000
CLAIMS_PER_CALL = 20      # one call for a typical answer; splits only long outliers.
                          # Lower values re-send the context more often, and the
                          # context dominates cost on a per-day token budget.


def _fit_output_budget(prompt: str, wanted: int) -> int:
    """Cap the output reservation at the model's real per-response limit."""
    return min(wanted, MODEL_MAX_TOKENS - OUTPUT_SAFETY_MARGIN)


def _ask(prompt: str, model: str, cache_dir: str, tag: str, max_tokens: int = 6000):
    """Run one cached judge call, returning the parsed JSON object."""
    cache = Path(cache_dir) / f"{tag}_{hashlib.sha256((model + prompt).encode()).hexdigest()[:20]}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    response = _client().chat.completions.create(
        model=model,
        max_tokens=_fit_output_budget(prompt, max_tokens),
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    data = _extract_json(response.choices[0].message.content or "")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def decompose_claims(answer: str, model: str = JUDGE_MODEL,
                     cache_dir: str = CACHE_DIR) -> list[str]:
    """Split an answer into atomic claims for groundedness checking."""
    data = _ask(DECOMPOSE_PROMPT.format(answer=answer), model, cache_dir, "decomp", MODEL_MAX_TOKENS - OUTPUT_SAFETY_MARGIN)
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

    # Verdicts are cheap per claim but the reasoning preamble is not: a 45-claim
    # answer left no output budget for the JSON after the model finished
    # thinking. Sub-batch so each call has room to answer, paying for the
    # context again only when an answer is unusually long.
    out: list[bool] = []
    for start in range(0, len(claims), CLAIMS_PER_CALL):
        group = claims[start:start + CLAIMS_PER_CALL]
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(group, 1))
        data = _ask(SUPPORT_BATCH_PROMPT.format(context=context, claims=numbered),
                    model, cache_dir, "support", max_tokens=MODEL_MAX_TOKENS - OUTPUT_SAFETY_MARGIN)
        verdicts = {int(r["n"]): r.get("verdict") == "supported"
                    for r in data if isinstance(r, dict) and "n" in r}
        out.extend(verdicts.get(i, False) for i in range(1, len(group) + 1))
    return out


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
        max_tokens=_fit_output_budget(
            PROMPT.format(claim=claim, answer=answer), 6000),
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
