"""Local, auditable index releases. No collection deletion, no implicit promotion.

Manifests and artifacts are write-once. Operational reports and the append-only
release journal are separate. The file lock serializes this project's CLI
operators; it is NOT a distributed lock against arbitrary Qdrant administrators.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from qdrant_client import models
from src.store import qdrant_store as qs

ROOT = Path(__file__).resolve().parents[2]
BUILDS = ROOT / "data/index_builds"
ACTIVE_ALIAS = "uw_manual_active"
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_once(path: Path, value):
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def build_dir(build_id: str, root=BUILDS) -> Path:
    if not NAME.fullmatch(build_id):
        raise ValueError("build id must be a lowercase safe identifier (1–64 characters)")
    return Path(root) / build_id


def read_build(build_id: str, root=BUILDS) -> dict:
    directory = build_dir(build_id, root)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["build_id"] != build_id:
        raise ValueError("manifest identity mismatch")
    for name, checksum in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents or digest(path.read_bytes()) != checksum:
            raise ValueError(f"artifact integrity failed: {name}")
    manifest["manifest_sha256"] = digest(raw)
    return manifest


def binding_for(build_id: str, root=BUILDS) -> dict:
    m = read_build(build_id, root)
    return {"build_id": build_id, "collection": m["collection"],
            "manifest_sha256": m["manifest_sha256"], "dense": m["dense"],
            "sparse": m["sparse"],
            "corpus_manifest_path": str(build_dir(build_id, root).resolve() / "corpus.json"),
            "build_root": str(Path(root).resolve())}


def same_index_content(left: dict, right: dict, root=BUILDS) -> bool:
    def corpus_contract(m):
        corpus = json.loads((build_dir(m["build_id"], root) / "corpus.json").read_text())
        corpus["documents"] = [{k:v for k,v in d.items() if k not in {"path", "caption_file"}}
                               for d in corpus["documents"]]
        assets = [{k: s[k] for k in ("source_doc", "kind", "sha256")} for s in m["sources"]]
        return {"corpus": corpus, "assets": assets}
    return (left["artifacts"]["points.json"] == right["artifacts"]["points.json"] and
            left["dense"] == right["dense"] and left["sparse"] == right["sparse"] and
            left["qdrant_schema"] == right["qdrant_schema"] and
            corpus_contract(left) == corpus_contract(right))


def validate_binding(binding: dict):
    expected = binding_for(binding["build_id"], binding.get("build_root", BUILDS))
    for field in ("collection", "manifest_sha256", "dense", "sparse", "corpus_manifest_path"):
        if binding.get(field) != expected[field]:
            raise ValueError(f"pinned index binding changed: {field}")
    if binding["sparse"]["model"] != "Qdrant/bm25":
        raise ValueError("this runtime only supports Qdrant/bm25 sparse query encoding")


def aliases(client) -> dict[str, str]:
    return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}


def resolve_active(client, alias=ACTIVE_ALIAS, root=BUILDS) -> dict:
    target = aliases(client).get(alias)
    if not target:
        raise ValueError(f"active alias {alias!r} is not configured; no silent fallback")
    matches = []
    for path in Path(root).glob("*/manifest.json"):
        m = json.loads(path.read_text())
        if m["collection"] == target:
            matches.append(m["build_id"])
    if len(matches) != 1:
        raise ValueError("alias must resolve to exactly one registered build")
    return binding_for(matches[0], root)


def exported_points(client, collection: str) -> list[dict]:
    result, offset = [], None
    while True:
        batch, offset = client.scroll(collection, limit=128, offset=offset,
                                      with_vectors=True, with_payload=True)
        result.extend({"id": str(p.id), "payload": p.payload, "vector": p.model_dump(mode="json")["vector"]} for p in batch)
        if offset is None:
            break
    return sorted(result, key=lambda p: p["id"])


def _validate_points(points: list[dict], dims: int):
    if not points:
        raise ValueError("empty build")
    ids = [p["payload"]["chunk_id"] for p in points]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids")
    for point in points:
        p, v = point["payload"], point["vector"]
        if point["id"] != qs.point_id(p["chunk_id"]) or not p.get("text"):
            raise ValueError("invalid chunk identity/text")
        dense, sparse = v[qs.DENSE], v[qs.SPARSE]
        if len(dense) != dims or not all(math.isfinite(x) for x in dense):
            raise ValueError("invalid dense vector")
        if len(sparse["indices"]) != len(sparse["values"]) or not all(math.isfinite(x) for x in sparse["values"]):
            raise ValueError("invalid sparse vector")


def _snapshot(build_id, points, corpus, dense, provenance, group_plan, *, root=BUILDS):
    """Capture exact installed payload/vectors; do not claim to recreate a past build."""
    directory = build_dir(build_id, root)
    _validate_points(points, dense["dims"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "assets").mkdir()
    corpus_doc = json.loads(Path(corpus).read_text())
    # Ensure scopes actually agree with the manifest, not just equal counts.
    specs = {d["source_doc"]: d for d in corpus_doc["documents"]}
    if {p["payload"]["source_doc"] for p in points} != set(specs):
        raise ValueError("manifest documents do not match collection documents")
    for point in points:
        payload = point["payload"]
        spec = specs[payload["source_doc"]]
        for key in ("lender_id", "document_id", "document_version", "version_status", "effective_from", "effective_to"):
            if payload.get(key) != spec.get(key):
                raise ValueError(f"scope mismatch: {payload['chunk_id']} / {key}")
    sources = []
    for item in corpus_doc["documents"]:
        for key in ("path", "caption_file"):
            if item.get(key):
                source = Path(item[key])
                source = source if source.is_absolute() else ROOT / source
                dest = directory / "assets" / source.name
                if dest.exists() and dest.read_bytes() != source.read_bytes():
                    raise ValueError("conflicting asset filenames")
                if not dest.exists():
                    shutil.copyfile(source, dest)
                sources.append({"source_doc": item["source_doc"], "kind": key,
                                "original_path": str(source), "sha256": digest(source.read_bytes())})
                item[key] = str(dest.resolve())
    write_once(directory / "corpus.json", corpus_doc)
    write_once(directory / "points.json", points)
    write_once(directory / "chunks.json", [p["payload"] for p in points])
    write_once(directory / "embedding_groups.json", group_plan)
    code = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for folder in ("src", "scripts")
            for p in sorted((ROOT / folder).rglob("*.py"))}
    write_once(directory / "code_hashes.json", code)
    shutil.copyfile(ROOT / "requirements.txt", directory / "requirements.txt")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    artifacts = {str(p.relative_to(directory)): digest(p.read_bytes())
                 for p in sorted(directory.rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "build_id": build_id,
                "collection": "uw_idx_" + build_id, "created_at": now(),
                "provenance": provenance, "point_count": len(points), "dense": dense,
                "sparse": {"model": "Qdrant/bm25", "modifier": "idf"},
                "qdrant_schema": {"dense_name": qs.DENSE, "distance": "Cosine", "sparse_name": qs.SPARSE},
                "sources": sources, "artifacts": artifacts,
                "git_head": git.stdout.strip() if git.returncode == 0 else None,
                "installed_packages": {name: importlib.metadata.version(name) for name in
                                       ("qdrant-client", "voyageai", "fastembed", "langgraph")}}
    write_once(directory / "manifest.json", manifest)
    return binding_for(build_id, root)


def capture_existing(client, collection, build_id, *, corpus=ROOT / "config/corpus_manifest.json", root=BUILDS):
    """Import observed vectors, without pretending to know their historical grouping."""
    if build_dir(build_id, root).exists():
        raise ValueError("build directory already exists; refusing overwrite")
    points = exported_points(client, collection)
    models_found = {p["payload"].get("embedding_model") for p in points}
    dims_found = {len(p["vector"][qs.DENSE]) for p in points}
    if len(models_found) != 1 or None in models_found or len(dims_found) != 1:
        raise ValueError("cannot determine a unique embedding model/dimension from payloads")
    dense = {"model": models_found.pop(), "dims": dims_found.pop(), "document_input_type": "document", "query_input_type": "query"}
    return _snapshot(build_id, points, corpus, dense,
                     {"kind": "observed_vector_import", "source_collection": collection,
                      "limitation": "Historical chunker and embedding groups not proven; code hashes describe capture-time code."},
                     {"status": "unknown_for_imported_vectors"}, root=root)


def prepare_build(build_id, chunks_path, *, corpus=ROOT / "config/corpus_manifest.json",
                  model="voyage-context-4", dims=1024, allow_provider=False, root=BUILDS):
    if build_dir(build_id, root).exists():
        raise ValueError("build directory already exists; refusing overwrite before provider calls")
    from src.embed import embedder, sparse
    chunks = json.loads(Path(chunks_path).read_text())
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["source_doc"], []).append(chunk)
    plans = []
    # Preflight all cache misses before making even one provider call.
    for doc, items in by_doc.items():
        groups = embedder._plan_groups([c["text"] for c in items], embedder.GROUP_TOKEN_BUDGET)
        cursor = 0
        for group in groups:
            plans.append({"source_doc": doc, "chunk_ids": [c["chunk_id"] for c in items[cursor:cursor + len(group)]],
                          "texts_sha256": digest(json_bytes(group)), "budget": embedder.GROUP_TOKEN_BUDGET})
            cursor += len(group)
            if not allow_provider and not embedder._is_cached(group, model, dims, "document", embedder.CACHE_DIR):
                raise ValueError("uncached embedding group; --allow-provider required to send document text to Voyage")
    points = []
    for items in by_doc.values():
        texts = [c["text"] for c in items]
        dense_vectors = embedder.embed_document(texts, model=model, dims=dims)
        sparse_vectors = sparse.embed_documents(texts)
        if len(dense_vectors) != len(items) or len(sparse_vectors) != len(items):
            raise ValueError("embedding count mismatch")
        for chunk, dense, sp in zip(items, dense_vectors, sparse_vectors):
            payload = {**chunk, "embedding_model": model, "embedding_dimension": dims,
                       "embedding_version": f"{model}:{dims}:document-context"}
            points.append({"id": qs.point_id(chunk["chunk_id"]), "payload": payload,
                           "vector": {qs.DENSE: dense, qs.SPARSE: sp.model_dump()}})
    return _snapshot(build_id, sorted(points, key=lambda p: p["id"]), corpus,
                     {"model": model, "dims": dims, "document_input_type": "document", "query_input_type": "query"},
                     {"kind": "embedded_chunk_artifact", "chunks_input_sha256": digest(Path(chunks_path).read_bytes()),
                      "limitation": "Chunker source captured at build time; supplied chunk artifact is authoritative."},
                     plans, root=root)


def install_candidate(client, build_id, root=BUILDS):
    m = read_build(build_id, root)
    collection = m["collection"]
    if client.collection_exists(collection) or collection in aliases(client):
        raise ValueError("candidate already exists; refusing overwrite (validate it or use a new build id)")
    qs.ensure_collection(client, m["dense"]["dims"], collection=collection)
    points = json.loads((build_dir(build_id, root) / "points.json").read_text())
    for start in range(0, len(points), 64):
        client.upsert(collection, points=[models.PointStruct(**p) for p in points[start:start+64]], wait=True)
    return validate_collection(client, build_id, root)


def validate_collection(client, build_id, root=BUILDS):
    m = read_build(build_id, root)
    expected = json.loads((build_dir(build_id, root) / "points.json").read_text())
    actual = exported_points(client, m["collection"])
    _validate_points(actual, m["dense"]["dims"])
    if [p["id"] for p in actual] != [p["id"] for p in expected]:
        raise ValueError("installed point identities differ")
    for got, want in zip(actual, expected):
        if got["payload"] != want["payload"] or got["vector"][qs.SPARSE]["indices"] != want["vector"][qs.SPARSE]["indices"]:
            raise ValueError("installed payload/sparse indices differ")
        for gv, wv in ((got["vector"][qs.DENSE], want["vector"][qs.DENSE]),
                       (got["vector"][qs.SPARSE]["values"], want["vector"][qs.SPARSE]["values"])):
            if len(gv) != len(wv) or any(not math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-6) for a,b in zip(gv,wv)):
                raise ValueError("installed vectors differ (beyond float32 normalization tolerance)")
    info = client.get_collection(m["collection"])
    dense = info.config.params.vectors[qs.DENSE]
    if dense.size != m["dense"]["dims"] or dense.distance != models.Distance.COSINE:
        raise ValueError("installed dense schema differs")
    if info.config.params.sparse_vectors[qs.SPARSE].modifier != models.Modifier.IDF:
        raise ValueError("installed sparse modifier differs")
    return {"build_id": build_id, "manifest_sha256": m["manifest_sha256"], "validated_at": now(),
            "point_count": len(actual), "integrity_passed": True}


@contextmanager
def admin_lock(root=BUILDS):
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / ".admin.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def journal(event, root=BUILDS):
    import os
    with (Path(root) / "releases.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": now(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def reconcile(client, operation_id: str, root=BUILDS):
    """Read current alias after a crash/timeout; never blindly replay a swap."""
    with admin_lock(root):
        records = [json.loads(line) for line in (Path(root) / "releases.jsonl").read_text().splitlines()]
        matches = [r for r in records if r["operation_id"] == operation_id]
        if not matches:
            raise ValueError("unknown operation")
        last = matches[-1]
        if last["event"] in {"completed", "not_applied"}:
            return last
        target = aliases(client).get(last["alias"])
        if target == last["to"]:
            status = "completed"
        elif target == last["from"]:
            status = "not_applied"
        else:
            raise ValueError("alias now points elsewhere; manual reconciliation required")
        outcome = {**last, "event": status, "reconciled": True, "time": now()}
        journal(outcome, root)
        return outcome


def check_quality_approval(path, report_bytes, report):
    """Explicit human gate for a changed build; never an automatic judge verdict."""
    approval_bytes = Path(path).read_bytes()
    approval = json.loads(approval_bytes)
    if (approval.get("evaluation_sha256") != digest(report_bytes) or
        approval.get("source_anchors_reviewed") is not True or
        approval.get("answer_quality_accepted") is not True or
        not str(approval.get("reviewer", "")).strip() or not str(approval.get("rationale", "")).strip()):
        raise ValueError("quality approval must bind the paired report and record both source/answer review")
    answer_path = Path(approval["answer_evaluation_path"])
    if not answer_path.is_absolute():
        answer_path = Path(path).resolve().parent / answer_path
    if digest(answer_path.read_bytes()) != approval.get("answer_evaluation_sha256"):
        raise ValueError("answer-quality evidence hash mismatch")
    # Review may be human-authored or a checked judge report; its provenance
    # must still explicitly name these exact builds.
    answers = json.loads(answer_path.read_text())
    for field in ("active_manifest_sha256", "candidate_manifest_sha256"):
        if answers.get(field) != report.get(field):
            raise ValueError("answer review belongs to different builds")
    if report.get("regressions") or not report["builds"][report["candidate_build"]]["mapping"]["all_groups_covered"]:
        raise ValueError("retrieval/structural regressions must be resolved before promotion")
    return digest(approval_bytes)


def release(client, build_id, *, expected_collection: str | None, reason: str,
            alias=ACTIVE_ALIAS, root=BUILDS, evaluation: str | None = None,
            quality_approval: str | None = None, bootstrap=False, rollback=False):
    """Atomic Qdrant alias swap with local serialized precondition and durable intent.

    Rollback can target only a previously released build. Promotion requires a
    paired report bound to these two manifests. Bootstrap is explicitly logged
    as unqualified, for establishing a baseline, not as passing an evaluation.
    """
    if not reason.strip() or not NAME.fullmatch(alias):
        raise ValueError("valid alias and nonempty release reason required")
    with admin_lock(root):
        prior = aliases(client).get(alias)
        if prior != expected_collection:
            raise ValueError(f"stale release precondition: expected {expected_collection}, actual {prior}")
        validation = validate_collection(client, build_id, root)
        target = read_build(build_id, root)["collection"]
        if target == prior:
            raise ValueError("alias already points to target")
        report_hash, approval_hash = None, None
        if bootstrap:
            if prior is not None or rollback:
                raise ValueError("bootstrap is only for an absent alias")
        elif rollback:
            records = [json.loads(line) for line in (Path(root) / "releases.jsonl").read_text().splitlines()]
            if not any(r.get("event") == "completed" and r.get("alias") == alias and r.get("to") == target for r in records):
                raise ValueError("rollback target was never released for this alias")
        else:
            if not evaluation:
                raise ValueError("promotion requires an active-versus-candidate evaluation report")
            raw = Path(evaluation).read_bytes()
            report = json.loads(raw)
            active = resolve_active(client, alias, root)
            if (report.get("active_manifest_sha256") != active["manifest_sha256"] or
                report.get("candidate_manifest_sha256") != validation["manifest_sha256"]):
                raise ValueError("evaluation belongs to different builds")
            if report.get("promotion_eligible") is not True:
                if not quality_approval:
                    raise ValueError("changed/ineligible build requires explicit source and answer-quality approval")
                approval_hash = check_quality_approval(quality_approval, raw, report)
            elif not same_index_content(read_build(active["build_id"], root), read_build(build_id, root), root):
                raise ValueError("automatic promotion is limited to identical-index controls")
            report_hash = digest(raw)
        op = str(uuid.uuid4())
        base = {"operation_id": op, "alias": alias, "from": prior, "to": target, "build_id": build_id,
                "reason": reason, "kind": "bootstrap" if bootstrap else "rollback" if rollback else "promotion",
                "manifest_sha256": validation["manifest_sha256"], "evaluation_sha256": report_hash,
                "quality_approval_sha256": approval_hash}
        journal({**base, "event": "intent"}, root)
        actions = []
        if prior:
            actions.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        actions.append(models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=alias, collection_name=target)))
        try:
            client.update_collection_aliases(change_aliases_operations=actions)
        except Exception:
            journal({**base, "event": "unknown_outcome", "instruction": "inspect alias before retrying"}, root)
            raise
        if aliases(client).get(alias) != target:
            journal({**base, "event": "unknown_outcome"}, root)
            raise RuntimeError("alias verification failed")
        journal({**base, "event": "completed"}, root)
        return base
