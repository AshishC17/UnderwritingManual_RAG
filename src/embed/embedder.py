"""Voyage embeddings with a disk cache.

Two model families are supported and they cache differently:

- Standard models (`voyage-4*`) embed each chunk independently.
- `voyage-context-4` embeds every chunk *in the context of its whole document*,
  so a chunk's vector depends on its neighbours. Caching those per chunk text
  would be wrong: edit one chunk and its neighbours' vectors go stale silently.

The cache is therefore keyed on the whole document's chunk set, which is correct
for both families. Re-embedding one document is cheap (the corpus is ~17K tokens
against a 200M free tier), so this trades a little redundant work for a cache
that cannot go quietly stale.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL = "voyage-context-4"
DIMS = 1024
CACHE_DIR = "data/interim/embeddings"

CONTEXTUAL_MODELS = {"voyage-context-4"}


class MissingCredentials(RuntimeError):
    pass


def _key(model: str, dims: int, input_type: str, texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{dims}|{input_type}|".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:20]


def _client():
    import voyageai

    if not os.environ.get("VOYAGE_API_KEY"):
        raise MissingCredentials(
            "VOYAGE_API_KEY is not set. Get a key at dash.voyageai.com "
            "(200M tokens free), then `export VOYAGE_API_KEY=...`"
        )
    return voyageai.Client()


def embed_document(
    texts: list[str],
    model: str = MODEL,
    dims: int = DIMS,
    cache_dir: str = CACHE_DIR,
) -> list[list[float]]:
    """Embed all chunks of one document, in order. Cached as a unit."""
    return _embed(texts, model, dims, "document", cache_dir)


def embed_query(
    text: str,
    model: str = MODEL,
    dims: int = DIMS,
    cache_dir: str = CACHE_DIR,
) -> list[float]:
    """Embed a single query. `input_type="query"` matters: Voyage encodes queries
    and documents differently, and mixing them measurably degrades retrieval."""
    return _embed([text], model, dims, "query", cache_dir)[0]


def _embed(
    texts: list[str],
    model: str,
    dims: int,
    input_type: str,
    cache_dir: str,
) -> list[list[float]]:
    cache = Path(cache_dir) / f"{_key(model, dims, input_type, texts)}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    client = _client()
    if model in CONTEXTUAL_MODELS:
        result = client.contextualized_embed(
            inputs=[texts],
            model=model,
            input_type=input_type,
            output_dimension=dims,
        )
        vectors = result.results[0].embeddings
    else:
        vectors = client.embed(
            texts, model=model, input_type=input_type, output_dimension=dims
        ).embeddings

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(vectors))
    return vectors


def count_tokens(texts: list[str], model: str = MODEL) -> int:
    """Token count from Voyage's own tokenizer, for cost and budget checks."""
    return _client().count_tokens(texts, model=model)
