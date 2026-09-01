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
import time
from pathlib import Path

MODEL = "voyage-context-4"
DIMS = 1024
CACHE_DIR = "data/interim/embeddings"

CONTEXTUAL_MODELS = {"voyage-context-4"}

# Voyage's free tier (no payment method on file) allows 10K tokens/min and 3
# requests/min. Stay under both, with margin. Raising GROUP_TOKEN_BUDGET once a
# payment method exists lets the whole document embed as a single contextual
# group — see `embed_document` for why that is better.
GROUP_TOKEN_BUDGET = 9000
REQUEST_SPACING_S = 62


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


def _plan_groups(texts: list[str], budget: int) -> list[list[str]]:
    """Split chunks into contiguous groups that each fit the token budget.

    Groups are balanced rather than greedily filled: with a contextual model each
    group is one contextualization unit, so two even groups leave every chunk with
    a similar amount of surrounding context, where a greedy split would leave the
    last group thin.
    """
    counts = [_approx_tokens(t) for t in texts]
    total = sum(counts)
    n_groups = max(1, -(-total // budget))
    if n_groups == 1:
        return [texts]

    target = total / n_groups
    groups: list[list[str]] = []
    current: list[str] = []
    running = 0
    for text, count in zip(texts, counts):
        if current and running + count > target and len(groups) < n_groups - 1:
            groups.append(current)
            current, running = [], 0
        current.append(text)
        running += count
    if current:
        groups.append(current)
    return groups


def _approx_tokens(text: str) -> int:
    """Cheap local estimate, only used to size groups."""
    return max(1, len(text) // 4)


def embed_document(
    texts: list[str],
    model: str = MODEL,
    dims: int = DIMS,
    cache_dir: str = CACHE_DIR,
    budget: int = GROUP_TOKEN_BUDGET,
    verbose: bool = True,
) -> list[list[float]]:
    """Embed all chunks of one document, in order.

    Ideally the whole document is one request, so a contextual model sees every
    chunk's full surroundings. Where the free-tier token/minute cap forces a
    split, each group is contextualized independently — chunks still get local
    context, just not document-wide. Groups are cached separately.
    """
    groups = _plan_groups(texts, budget)
    if verbose and len(groups) > 1:
        sizes = [sum(_approx_tokens(t) for t in g) for g in groups]
        print(
            f"  splitting into {len(groups)} contextualization groups "
            f"(~{sizes} tokens) to stay under the {budget}/min free-tier cap"
        )

    vectors: list[list[float]] = []
    for i, group in enumerate(groups):
        if i and not _is_cached(group, model, dims, "document", cache_dir):
            if verbose:
                print(f"  waiting {REQUEST_SPACING_S}s for the rate-limit window...")
            time.sleep(REQUEST_SPACING_S)
        vectors.extend(_embed(group, model, dims, "document", cache_dir))
        if verbose:
            print(f"  group {i + 1}/{len(groups)}: {len(group)} chunks embedded")
    return vectors


def embed_query(
    text: str,
    model: str = MODEL,
    dims: int = DIMS,
    cache_dir: str = CACHE_DIR,
) -> list[float]:
    """Embed a single query. `input_type="query"` matters: Voyage encodes queries
    and documents differently, and mixing them measurably degrades retrieval."""
    return _embed([text], model, dims, "query", cache_dir)[0]


def _cache_file(
    texts: list[str], model: str, dims: int, input_type: str, cache_dir: str
) -> Path:
    return Path(cache_dir) / f"{_key(model, dims, input_type, texts)}.json"


def _is_cached(
    texts: list[str], model: str, dims: int, input_type: str, cache_dir: str
) -> bool:
    return _cache_file(texts, model, dims, input_type, cache_dir).exists()


def _embed(
    texts: list[str],
    model: str,
    dims: int,
    input_type: str,
    cache_dir: str,
) -> list[list[float]]:
    cache = _cache_file(texts, model, dims, input_type, cache_dir)
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
