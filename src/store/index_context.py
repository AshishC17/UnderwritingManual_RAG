"""Per-node index binding. ContextVar isolates concurrent requests, not globals.

The serializable binding lives in graph state/checkpoints. This short-lived
context only propagates it to existing retrieval, resolver and embedding APIs.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_binding = ContextVar("index_binding", default=None)
ROOT = Path(__file__).resolve().parents[2]


def current() -> dict:
    return _binding.get() or {}


@contextmanager
def bind(binding: dict | None):
    token = _binding.set(binding or None)
    try:
        yield
    finally:
        _binding.reset(token)


def collection_name() -> str:
    return current().get("collection", "uw_manual")


def corpus_path(path) -> Path:
    requested = Path(path)
    absolute = requested if requested.is_absolute() else ROOT / requested
    if absolute.resolve() == (ROOT / "config/corpus_manifest.json").resolve():
        return Path(current().get("corpus_manifest_path", absolute))
    return absolute
