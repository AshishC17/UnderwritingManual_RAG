"""Query decomposition: the combined gate-and-split step.

Runs before retrieval and sees only the raw question, never retrieved context —
that ordering is what makes decompose-then-retrieve possible in the first place.

A single call does both jobs: decide whether the question asks more than a
couple of distinct things, and if so, split it. An earlier design used a free
regex heuristic (clause/keyword counting) as a gate in front of this call, but
that heuristic was fit to patterns in a small, self-authored set of eval
questions — exactly the kind of hard-fitted, non-generalizing rule this project
has repeatedly flagged as a trap elsewhere. Real questions from real users won't
match a hand-built keyword list. Judging complexity is a semantic call, so an
LLM makes it, on every question, rather than a heuristic pre-filter.

Model is qwen3.8-27b — the *original* judge model (kappa 0.900, no reasoning-
block or output-budget surprises encountered in this project) — not a new,
untested model. The real constraint was never "must differ from the judge," it
was "must differ from the generator" (cost mismatch, and the generator only
runs after retrieval exists anyway). Reusing a model whose quirks are already
known avoids repeating the qwen3.6-27b discovery cost from the groundedness work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from langsmith import traceable

MODEL = "qwen/qwen3.8-27b"
CACHE_DIR = "data/interim/decompositions"
MAX_SUB_QUERIES = 3

PROMPT = """You are preparing a question for a fact-lookup system.

Your DEFAULT action is to leave the question completely unchanged. Splitting is \
a rare exception, not the normal outcome. Most questions must be returned \
verbatim.

QUESTION:
{question}

Count how many SEPARATE FACT LOOKUPS this question requires. A separate lookup \
means a distinct fact that must be found independently, in a different place. \
The following do NOT count as separate lookups:
- A second clause about the same fact.
- A qualifier, condition, or detail attached to a fact.
- Asking for two attributes of one thing (e.g. a value and its source).
- A grammatical "and" or "or" joining parts of one idea.

Then apply this rule strictly:
- Count is 1 or 2 -> output the QUESTION unchanged, verbatim, as the ONLY item \
in the array. Do not split it. Do not reword it. Do not merge or re-punctuate \
it. Return the exact original characters.
- Count is 3 or more -> split into at most 3 standalone sub-questions, each of \
which must be answerable entirely on its own, must preserve every named entity, \
code, and term verbatim (keep "Code 120" as "Code 120"), must together cover \
everything the original asks, and must introduce nothing new.

Reply with only a JSON array of strings, no other text."""


class MissingCredentials(RuntimeError):
    pass


def _key(model: str, question: str) -> str:
    """Keyed on the PROMPT too, not just model+question.

    Without the prompt in the key, editing the instructions silently returns
    results generated under the old prompt — which would make prompt-compliance
    testing meaningless, since every change would appear to have no effect.
    """
    return hashlib.sha256(f"{model}|{PROMPT}|{question}".encode()).hexdigest()[:20]


def _client():
    import groq

    if not os.environ.get("GROQ_API_KEY"):
        raise MissingCredentials("GROQ_API_KEY is not set")
    return groq.Groq()


def _extract_json(text: str):
    """Same defensive extraction as src/eval/judge.py: strip a <think> block if
    present (qwen3.8 hasn't shown this behavior in this project, but the check
    is free and the project has already been burned once by assuming a model
    "probably" doesn't reason before answering). Raise on anything unparseable
    rather than silently falling back to "no split" — a swallowed parse error
    here would silently disable decomposition for that question."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.S)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned).strip()
    start = cleaned.find("[")
    if start == -1:
        raise ValueError(f"no JSON array in decomposition reply: {text[:120]!r}")
    return json.loads(cleaned[start:])


@traceable(run_type="llm", name="decompose_query (qwen3.8 gate+split)")
def decompose_query(
    question: str,
    model: str = MODEL,
    cache_dir: str = CACHE_DIR,
) -> list[str]:
    """[question] unchanged if it asks <=2 things; up to MAX_SUB_QUERIES
    standalone sub-questions if it asks more."""
    cache = Path(cache_dir) / f"{_key(model, question)}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    response = _client().chat.completions.create(
        model=model,
        max_tokens=2000,
        temperature=0,  # deterministic: same question must decompose the same way
        messages=[{"role": "user", "content": PROMPT.format(question=question)}],
    )
    data = _extract_json(response.choices[0].message.content or "")
    sub_queries = [str(q) for q in data] if isinstance(data, list) and data else [question]

    if len(sub_queries) > MAX_SUB_QUERIES:
        print(f"  WARNING: decomposition returned {len(sub_queries)} sub-queries "
              f"(cap is {MAX_SUB_QUERIES}), truncating: {question[:80]!r}")
        sub_queries = sub_queries[:MAX_SUB_QUERIES]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sub_queries))
    return sub_queries
