"""Validated corpus manifest and cross-version chunk lineage helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .chunker import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_STATUSES = {"current", "historical"}


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid RFC3339 timestamp: {value!r}") from exc


@dataclass(frozen=True)
class DocumentSpec:
    path: str
    source_doc: str
    lender_id: str
    lender_name: str
    product_id: str
    product_name: str
    document_family: str
    document_id: str
    document_version: str
    version_order: int
    effective_from: str
    effective_to: str | None
    version_status: str
    supersedes_version: str | None
    authority_level: str
    chunking_version: str
    caption_file: str | None = None

    def resolved_path(self, root: Path = PROJECT_ROOT) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else root / path

    def resolved_caption_file(self, root: Path = PROJECT_ROOT) -> Path | None:
        if not self.caption_file:
            return None
        path = Path(self.caption_file)
        return path if path.is_absolute() else root / path

    def chunk_metadata(self) -> dict[str, Any]:
        fields = (
            "lender_id",
            "lender_name",
            "product_id",
            "product_name",
            "document_family",
            "document_id",
            "document_version",
            "version_order",
            "effective_from",
            "effective_to",
            "version_status",
            "supersedes_version",
            "authority_level",
            "chunking_version",
        )
        return {name: getattr(self, name) for name in fields}


def load_manifest(
    path: str | Path,
    *,
    root: Path = PROJECT_ROOT,
    require_files: bool = True,
) -> list[DocumentSpec]:
    """Load and validate document identity plus non-overlapping effective dates."""
    from src.store.index_context import corpus_path
    manifest_path = corpus_path(path) if root == PROJECT_ROOT else Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    raw = json.loads(manifest_path.read_text())
    chunking_version = raw.get("chunking_version")
    if not chunking_version:
        raise ValueError("manifest requires chunking_version")

    specs: list[DocumentSpec] = []
    for item in raw.get("documents", []):
        specs.append(
            DocumentSpec(
                **item,
                chunking_version=chunking_version,
            )
        )
    if not specs:
        raise ValueError("manifest must contain at least one document")

    source_docs: set[str] = set()
    versions: set[tuple[str, str]] = set()
    by_document: dict[str, list[DocumentSpec]] = defaultdict(list)
    for spec in specs:
        if spec.source_doc != Path(spec.path).name:
            raise ValueError(
                f"source_doc {spec.source_doc!r} must equal file name {Path(spec.path).name!r}"
            )
        if spec.source_doc in source_docs:
            raise ValueError(f"duplicate source_doc: {spec.source_doc}")
        source_docs.add(spec.source_doc)
        identity = (spec.document_id, spec.document_version)
        if identity in versions:
            raise ValueError(f"duplicate document version: {identity}")
        versions.add(identity)
        if spec.version_status not in VALID_STATUSES:
            raise ValueError(
                f"invalid version_status {spec.version_status!r} for {spec.source_doc}"
            )
        start = _parse_datetime(spec.effective_from)
        end = _parse_datetime(spec.effective_to) if spec.effective_to else None
        if end is not None and start > end:
            raise ValueError(f"effective_from is after effective_to: {spec.source_doc}")
        if spec.version_status == "current" and end is not None:
            raise ValueError(f"current document must have null effective_to: {spec.source_doc}")
        if spec.version_status == "historical" and end is None:
            raise ValueError(f"historical document needs effective_to: {spec.source_doc}")
        if require_files and not spec.resolved_path(root).is_file():
            raise FileNotFoundError(spec.resolved_path(root))
        caption_file = spec.resolved_caption_file(root)
        if require_files and caption_file is not None and not caption_file.is_file():
            raise FileNotFoundError(caption_file)
        by_document[spec.document_id].append(spec)

    for document_id, family in by_document.items():
        ordered = sorted(family, key=lambda spec: spec.version_order)
        if len({spec.version_order for spec in ordered}) != len(ordered):
            raise ValueError(f"duplicate version_order for {document_id}")
        if sum(spec.version_status == "current" for spec in ordered) != 1:
            raise ValueError(f"{document_id} must have exactly one current version")
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = _parse_datetime(previous.effective_to) if previous.effective_to else None
            current_start = _parse_datetime(current.effective_from)
            if previous_end is None or previous_end >= current_start:
                raise ValueError(
                    f"effective periods overlap or are unordered: "
                    f"{previous.source_doc}, {current.source_doc}"
                )
            if current.supersedes_version != previous.document_version:
                raise ValueError(
                    f"{current.source_doc} must supersede {previous.document_version}"
                )

    return specs


def assign_chunk_revisions(chunks: list[Chunk]) -> None:
    """Assign revisions by comparing content hashes along each structural lineage."""
    by_lineage: dict[str, list[Chunk]] = defaultdict(list)
    for item in chunks:
        if not item.lineage_id or item.version_order is None or not item.content_hash:
            raise ValueError(f"chunk lacks lineage metadata: {item.chunk_id}")
        by_lineage[item.lineage_id].append(item)

    for lineage in by_lineage.values():
        ordered = sorted(lineage, key=lambda item: item.version_order or 0)
        revision = 0
        previous_hash: str | None = None
        for item in ordered:
            if item.content_hash != previous_hash:
                revision += 1
                previous_hash = item.content_hash
            item.chunk_revision = revision
