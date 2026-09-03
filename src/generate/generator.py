"""Answer generation over retrieved chunks.

The system prompt encodes *general* properties of this corpus — reused rule codes,
narrow exceptions, the gap between a control passing and an application being
approved. It deliberately contains no case-specific instructions: telling the
model what to say about code 120 would tune the prompt to the eval set rather
than to the document, and the gain would not survive a new question.

Answers are cached on (model, question, context) so re-running the eval after a
scoring change costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL = "claude-opus-5"
CACHE_DIR = "data/interim/generations"
MAX_TOKENS = 4000

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


class MissingCredentials(RuntimeError):
    pass


def _key(model: str, question: str, context: str) -> str:
    h = hashlib.sha256(f"{model}|{question}|{context}".encode())
    return h.hexdigest()[:20]


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


def _client():
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise MissingCredentials(
            "ANTHROPIC_API_KEY is not set. Get one at console.anthropic.com "
            "(billed separately from a Claude Code subscription) and add it to .env"
        )
    return anthropic.Anthropic()


def generate(
    question: str,
    chunks: list[dict],
    model: str = MODEL,
    cache_dir: str = CACHE_DIR,
) -> str:
    context = build_context(chunks)
    cache = Path(cache_dir) / f"{_key(model, question, context)}.json"
    if cache.exists():
        return json.loads(cache.read_text())["answer"]

    response = _client().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {question}",
        }],
    )
    answer = "".join(b.text for b in response.content if b.type == "text").strip()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "answer": answer,
        "model": model,
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }))
    return answer
