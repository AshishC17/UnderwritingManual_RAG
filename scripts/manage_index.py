"""Index lifecycle CLI; all operations explicit, never delete collections.

Examples: python scripts/manage_index.py --help
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from src.store import index_builds as ib
from src.store import qdrant_store as qs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--root", type=Path, default=ib.BUILDS)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture", help="snapshot existing payloads/vectors; not a claim of historical reproducibility")
    p.add_argument("build_id")
    p.add_argument("--collection", default=qs.COLLECTION)
    p.add_argument("--corpus", type=Path, default=ROOT / "config/corpus_manifest.json")
    p = sub.add_parser("prepare", help="freeze chunk artifacts + embeddings; no Qdrant writes")
    p.add_argument("build_id")
    p.add_argument("--chunks", required=True, type=Path)
    p.add_argument("--corpus", type=Path, default=ROOT / "config/corpus_manifest.json")
    p.add_argument("--model", default="voyage-context-4")
    p.add_argument("--dims", default=1024, type=int)
    p.add_argument("--allow-provider", action="store_true")
    for name in ("install", "validate", "inspect"):
        p = sub.add_parser(name)
        p.add_argument("build_id")
    sub.add_parser("status", help="alias destinations + unresolved journal intents")
    p = sub.add_parser("reconcile", help="resolve a recorded timeout/crash by reading the alias, without replaying writes")
    p.add_argument("operation_id")
    for name in ("bootstrap", "promote", "rollback"):
        p = sub.add_parser(name)
        p.add_argument("build_id")
        p.add_argument("--alias", default=ib.ACTIVE_ALIAS)
        p.add_argument("--reason", required=True)
        if name != "bootstrap":
            p.add_argument("--expected", required=True, help="current concrete collection; stale requests are rejected")
        if name == "promote":
            p.add_argument("--evaluation", required=True)
            p.add_argument("--quality-approval", help="human source+answer review required for changed indexes")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    if a.command == "prepare":
        result = ib.prepare_build(a.build_id, a.chunks, corpus=a.corpus, model=a.model, dims=a.dims,
                                  allow_provider=a.allow_provider, root=a.root)
    elif a.command == "inspect":
        result = ib.read_build(a.build_id, a.root)
    else:
        with closing(qs.connect(a.url)) as client:
            if a.command == "capture":
                result = ib.capture_existing(client, a.collection, a.build_id, corpus=a.corpus, root=a.root)
            elif a.command == "install":
                result = ib.install_candidate(client, a.build_id, a.root)
            elif a.command == "validate":
                result = ib.validate_collection(client, a.build_id, a.root)
            elif a.command == "reconcile":
                result = ib.reconcile(client, a.operation_id, a.root)
            elif a.command == "status":
                path = a.root / "releases.jsonl"
                events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
                latest = {e["operation_id"]: e for e in events}
                result = {"aliases": ib.aliases(client), "unresolved_operations": [e for e in latest.values() if e["event"] not in {"completed", "not_applied"}]}
            else:
                result = ib.release(client, a.build_id, expected_collection=getattr(a, "expected", None),
                                    reason=a.reason, alias=a.alias, root=a.root,
                                    evaluation=getattr(a, "evaluation", None),
                                    quality_approval=getattr(a, "quality_approval", None),
                                    bootstrap=a.command == "bootstrap", rollback=a.command == "rollback")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
