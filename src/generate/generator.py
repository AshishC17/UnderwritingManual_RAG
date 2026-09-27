"""Answer generation over retrieved chunks.

The system prompt encodes *general* properties of this corpus — reused rule codes,
narrow exceptions, the gap between a control passing and an application being
approved. It deliberately contains no case-specific instructions: telling the
model what to say about code 120 would tune the prompt to the eval set rather
than to the document, and the gain would not survive a new question.

Answers are cached on (model, system prompt, question, context), so scoring-only
changes can reuse generation while prompt/evidence changes invalidate it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from src.guardrails.privacy import cache_json, cache_read, sanitize

from src.guardrails.tracing import traceable
from src.util.telemetry import cache_hit, tracked_call

MODEL = "openai/gpt-oss-120b"
CACHE_DIR = "data/interim/generations"
MAX_TOKENS = 4000
REUSE_CACHE_DIR = "data/interim/reuse_generations"

SYSTEM = """You answer questions about an underwriting policy manual using only \
the excerpts provided.

Rules:
- Use only the provided context. If it does not contain the answer, say so
  explicitly rather than inferring or drawing on outside knowledge.
- Cite the chunk_id for every factual claim, as [chunk_id].
- State conditions exactly as written. Never broaden a conditional rule into a
  general one: if an exception requires two conditions, say that both are
  required.
- Rule codes are reused across stages with different meanings. When citing a
  code, state which table or stage it came from.
- A rule passing is not the same as an application being approved. Do not
  conflate an intermediate control outcome with a final decision.
- If a footnote or exception is attached to a rule in the context, include it.
  Omitting an exception is an error.
- Be concise. Answer the question asked, without restating the context."""


COMPARISON_SYSTEM = SYSTEM + """

This request compares two separately scoped policy manuals. Evidence is grouped
under locked lender/version headings. Keep the groups independent:
- Compare the same requested dimension on both sides where evidence permits.
- Never use one lender's evidence to fill a missing fact for the other lender.
- Different terms or workflow stages are not automatically equivalent; explain
  the distinction instead of forcing a match.
- Attribute every factual claim to the correct lender/version and exact chunk id.
- If one side is not established by its supplied evidence, state that scoped
  limitation rather than making a corpus-wide absence claim.
"""


REUSE_SYSTEM = """You are revising your own previous answer about an \
underwriting policy manual. You are given three things, and they are not \
equally trustworthy:

- ORIGINAL QUESTION: the scenario as the user actually described it. Trust \
this as the given premise — it is not a claim to verify, it is the situation \
being asked about. A policy manual states general rules; it will not restate \
scenario-specific facts like "both of these fired at once" — that fact lives \
only in this question, so losing it here is how a later rule gets wrongly \
treated as skippable when it was not later at all.
- EVIDENCE: the policy excerpts you used before. This is the source of truth \
for what the rules ARE.
- PREVIOUS ANSWER: what you said last time. Its wording may be wrong; do not \
treat it as authoritative just because you said it before.

Follow this order strictly:

1. Verify. Check every factual claim in the PREVIOUS ANSWER against the \
EVIDENCE, read in light of the ORIGINAL QUESTION's actual scenario. Common \
error: a rule the scenario says already fired is not the same as work that \
was genuinely never executed — do not relabel an already-fired result as \
"skipped," "not evaluated," or "omitted" unless the scenario or evidence \
actually supports that.
2. Correct silently where needed. If a claim is not supported, restate it \
correctly. Do not flag every claim you checked — only call out a correction \
if the previous answer asserted something the evidence or scenario \
contradicts, not merely something it left unstated.
3. Apply the REQUEST. Once the facts are right, format the answer to satisfy \
the user's request below — simplify it, shorten it, explain a citation, \
whatever it asks. The request controls presentation, never the facts.

Cite the chunk_id for every factual claim, as [chunk_id], exactly as the \
previous answer should have.

Return JSON only, no prose:
{"answer": "<the revised answer, satisfying the request>",
"corrected_prior_claim": <true if the previous answer asserted something the \
evidence or scenario did not support and you fixed it, else false>}"""


class MissingCredentials(RuntimeError):
    pass


def _key(model: str, question: str, context: str) -> str:
    h = hashlib.sha256(f"{model}|{SYSTEM}|{question}|{context}".encode())
    return h.hexdigest()[:20]


def _reuse_key(
    model: str, instruction: str, prior_answer: str, original_question: str, chunk_ids: list[str],
    context: str = "",
) -> str:
    """Every input is either the user's raw text or stored data — never text an
    LLM generated this turn. Two identical requests against the same stored
    turn always hit the same cache entry, and always compute the same way."""
    h = hashlib.sha256()
    h.update(f"{model}|{REUSE_SYSTEM}|{instruction}|{prior_answer}|{original_question}|{context}|".encode())
    for cid in chunk_ids:
        h.update(cid.encode())
        h.update(b"\x00")
    return h.hexdigest()[:20]


def _extract_json(text: str) -> dict:
    """Same defensive extraction used across the project's other JSON-output
    calls (judge.py, decomposer.py, followup.py) — strip a <think> block if
    present, strip code fences, take the first balanced object, raise rather
    than silently falling back."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.S)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in reuse-generation reply: {text[:120]!r}")
    return json.loads(cleaned[start:])


def build_context(chunks: list[dict]) -> str:
    """Render retrieved chunks for the prompt, most relevant first.

    Each chunk carries its id and section so the model can cite precisely and can
    tell which stage a reused code came from.
    """
    parts = []
    for c in chunks:
        label = c.get("table_name") or c.get("section", "")
        parts.append(f"[{c['chunk_id']}] ({label})\n{c['text']}")
    return "\n\n".join(parts)


def build_comparison_context(scopes: list[dict], contexts: dict[str, list[dict]]) -> str:
    """Render evidence under explicit source boundaries for the comparison model."""
    from src.resolve.scope import scope_key

    sections = []
    for scope in scopes:
        key = scope_key(scope)
        title = f"EVIDENCE FOR {scope['lender_name']} {scope['document_version']} (scope_key={key})"
        sections.append(f"{title}\n\n{build_context(contexts.get(key, [])) or '(no evidence supplied)'}")
    return "\n\n---\n\n".join(sections)


def _client():
    import groq

    if not os.environ.get("GROQ_API_KEY"):
        raise MissingCredentials(
            "GROQ_API_KEY is not set. Get one at console.groq.com and add it to .env"
        )
    return groq.Groq()


@traceable(run_type="llm", name="generate (gpt-oss-120b)")
def generate(
    question: str,
    chunks: list[dict],
    model: str = MODEL,
    cache_dir: str = CACHE_DIR,
) -> str:
    context = build_context(chunks)
    cache = Path(cache_dir) / f"{_key(model, question, context)}.json"
    if cache.exists():
        cache_hit("groq", "generate", model)
        return cache_read(cache.read_text())["answer"]

    response = tracked_call("groq", "generate", model, _client().chat.completions.create,
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=0,  # deterministic: eval numbers must not wander between runs
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(cache_json({
        "answer": answer,
        "model": model,
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }))
    return answer


@traceable(run_type="llm", name="generate_comparison (gpt-oss-120b)")
def generate_comparison(
    question: str,
    scopes: list[dict],
    contexts: dict[str, list[dict]],
    model: str = MODEL,
    cache_dir: str = CACHE_DIR,
) -> str:
    context = build_comparison_context(scopes, contexts)
    key = hashlib.sha256(
        f"{model}|{COMPARISON_SYSTEM}|{question}|{context}".encode()
    ).hexdigest()[:20]
    cache = Path(cache_dir) / f"comparison-{key}.json"
    if cache.exists():
        cache_hit("groq", "generate_comparison", model)
        return cache_read(cache.read_text())["answer"]

    response = tracked_call(
        "groq", "generate_comparison", model, _client().chat.completions.create,
        model=model, max_tokens=MAX_TOKENS, temperature=0,
        messages=[
            {"role": "system", "content": COMPARISON_SYSTEM},
            {"role": "user", "content": f"Comparison evidence:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(cache_json({
        "answer": answer,
        "model": model,
        "chunk_ids_by_scope": {
            key: [chunk["chunk_id"] for chunk in chunks]
            for key, chunks in contexts.items()
        },
    }))
    return answer


@traceable(run_type="llm", name="revise_comparison (gpt-oss-120b)")
def revise_comparison_answer(
    request: str,
    draft: str,
    scopes: list[dict],
    contexts: dict[str, list[dict]],
    feedback: dict,
    model: str = MODEL,
    cache_dir: str = "data/interim/answer_revisions",
) -> str:
    context = build_comparison_context(scopes, contexts)
    system = COMPARISON_SYSTEM + """
You are correcting a comparison draft. REVIEW is diagnostic feedback, not policy
evidence. Correct supported omissions and attribution errors; remove unsupported
claims. Preserve each source boundary and state any unresolved side explicitly.
"""
    prompt = json.dumps({
        "request": request,
        "comparison_evidence": context,
        "draft": draft,
        "review": feedback,
    }, ensure_ascii=False)
    key = hashlib.sha256(json.dumps([model, system, prompt]).encode()).hexdigest()[:20]
    cache = Path(cache_dir) / f"comparison-{key}.json"
    if cache.exists():
        cache_hit("groq", "revise_comparison", model)
        return cache_read(cache.read_text())["answer"]
    response = tracked_call(
        "groq", "revise_comparison", model, _client().chat.completions.create,
        model=model, max_tokens=MAX_TOKENS, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    answer = (response.choices[0].message.content or "").strip()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(cache_json({"answer": answer, "model": model}))
    return answer


@traceable(run_type="llm", name="regenerate_from_evidence (gpt-oss-120b)")
def regenerate_from_evidence(
    instruction: str,
    prior_answer: str,
    chunks: list[dict],
    original_question: str = "",
    model: str = MODEL,
    cache_dir: str = REUSE_CACHE_DIR,
) -> dict:
    """Answer a transform/inspect follow-up from stored evidence, verifying
    the prior answer rather than assuming or blindly editing it.

    Unlike `generate`, this never receives a question composed this turn —
    `instruction` is the user's own words. No LLM output feeds this call's
    cache key, so two identical requests against the same stored turn always
    produce the same result: nothing left in the causal chain can vary.

    `original_question` carries the scenario as the user first stated it —
    separate from `instruction` (this turn's request) and `prior_answer`
    (untrusted). A policy manual states general rules, not scenario-specific
    facts like "these two fired at once"; that fact exists only in the
    original question, so without it here a rule the scenario said already
    fired can get misread as later, skippable work.

    Returns `{"answer": str, "corrected_prior_claim": bool}`. The correction
    flag is discovered here, not predicted upstream — whether the prior
    answer needed fixing is only knowable once it has actually been checked
    against evidence.

    KNOWN GAP, logged rather than silently patched around (see
    eval/RAG_failure_playbook, difficulty-ladder row 28): a scenario fact
    stated only in `original_question` (e.g. "these two fired at the same
    time") can still lose to a true *general* rule in `chunks` (e.g. "a later
    rule may be skipped after a terminal result") — the model applies the
    general rule to a case the scenario already said it does not cover.
    Passing `original_question` explicitly helped but did not close this;
    swapping to qwen/qwen3.8-27b did not either (same failure, plus a false
    self-reported correction on top). Untried: forcing the model to state
    which facts override which rules as its own field before answering.
    """
    context = build_context(chunks)
    chunk_ids = [c["chunk_id"] for c in chunks]
    cache = Path(cache_dir) / (
        f"{_reuse_key(model, instruction, prior_answer, original_question, chunk_ids, context)}.json"
    )
    if cache.exists():
        cache_hit("groq", "reuse_generation", model)
        return cache_read(cache.read_text())

    response = tracked_call("groq", "reuse_generation", model, _client().chat.completions.create,
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=0,
        messages=[
            {"role": "system", "content": REUSE_SYSTEM},
            {"role": "user", "content": (
                f"ORIGINAL QUESTION:\n{original_question}\n\n"
                f"EVIDENCE:\n\n{context}\n\n"
                f"PREVIOUS ANSWER:\n{prior_answer}\n\n"
                f"REQUEST: {instruction}"
            )},
        ],
    )
    data = _extract_json(response.choices[0].message.content or "")
    result = {
        "answer": str(data.get("answer") or "").strip(),
        "corrected_prior_claim": bool(data.get("corrected_prior_claim")),
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(cache_json(result))
    return result


REVISION_SYSTEM = SYSTEM + """
You are correcting a DRAFT, not treating it as policy. Address the original
request and explicit scenario facts, using only the supplied policy evidence.
REVIEW is fallible diagnostic feedback, not an additional factual source.
Correct all identified omissions/contradictions that evidence establishes;
remove unsupported embellishments. If a requested fact remains unavailable,
state that limitation. Do not invent support for a prior assertion. Preserve
the user's requested format where compatible with correctness.
"""


@traceable(run_type="llm", name="revise_answer (gpt-oss-120b)")
def revise_answer(request: str, draft: str, chunks: list[dict], feedback: dict,
                  model: str = MODEL, cache_dir: str = "data/interim/answer_revisions") -> str:
    context = build_context(chunks)
    prompt = json.dumps({"request": request, "evidence": context, "draft": draft, "review": feedback}, ensure_ascii=False)
    key = hashlib.sha256(json.dumps([model, REVISION_SYSTEM, prompt]).encode()).hexdigest()[:20]
    cache = Path(cache_dir) / f"{key}.json"
    if cache.exists():
        cache_hit("groq", "revise_answer", model)
        return cache_read(cache.read_text())["answer"]
    response = tracked_call("groq", "revise_answer", model, _client().chat.completions.create,
                            model=model, max_tokens=MAX_TOKENS, temperature=0,
                            messages=[{"role": "system", "content": REVISION_SYSTEM}, {"role": "user", "content": prompt}])
    answer = (response.choices[0].message.content or "").strip()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(cache_json({"answer": answer, "model": model}))
    return answer
