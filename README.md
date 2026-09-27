# Underwriting Manual RAG

A learning project for answering questions over synthetic underwriting policies:
two fictional lenders, two document versions each. It explores retrieval quality,
LangGraph orchestration, evaluation and reliable execution through a local web app.

This is a document assistant, not a loan decision engine or a production deployment.
The experimental semantic warehouse and text-to-SQL work are outside this release.

## What you can explore

- Structure-aware prose/table chunking and textual flowchart captions.
- Voyage contextual embeddings, Qdrant dense + BM25 retrieval, RRF and reranking.
- Lender/version resolution, follow-up interpretation and separate retrieval for
  each side of a two-lender comparison.
- Bounded review that can revise an answer or search for missing evidence.
- Input/evidence/output checks, citations and optional LangSmith tracing.
- PostgreSQL checkpoints/transcripts, request deduplication/recovery code, and
  opt-in Qdrant index promotion, rollback and per-turn pinning.

Exact caches and an experimental query-embedding semantic cache are included.
Its similarity threshold is not a calibrated guarantee of equivalent intent.

## Run locally

Use Python 3.12 (used for local testing), Git and Docker Compose. Run commands
from the repository root. Python runs the app; Compose starts only Qdrant and
PostgreSQL. Ports 8000, 6333/6334 and 5432 must be available.

```bash
git clone https://github.com/AshishC17/UnderwritingManual_RAG.git
cd UnderwritingManual_RAG
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill in `GROQ_API_KEY` and `VOYAGE_API_KEY` in `.env`. LangSmith is optional and
disabled in the example. Generation, review, semantic guards, embeddings and
reranking use paid/rate-limited provider APIs. Confirm access to configured models
in your provider account before running the live demo.

```bash
docker compose up -d qdrant postgres
python scripts/setup_state_db.py
python scripts/run_embed.py --chunks data/processed/chunks_v2.json
python -m uvicorn src.app.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Embedding populates a fresh `uw_manual` collection
from the included four-document chunk snapshot. Public local embedding/tokenizer
assets also download on first use. Caches and database storage stay local.

Try these in the same conversation:

1. “For NovaCred Financial, what happens when a credit freeze is found at prescreen?”
2. “Explain that in simpler words.”
3. “Compare that with LumenTrail's current policy.”

Stop the foreground app with Ctrl-C, then rerun uvicorn to examine persistence.
Do not start a second process on port 8000. `docker compose stop` stops the
databases without deleting their bind-mounted storage.

## Corpus and indexing

`config/corpus_manifest.json` defines document identity, dates and versions.
The four PDFs, captions and `chunks_v2.json` are reproducibility fixtures;
`chunks.json` supports legacy single-manual evaluations. The synthetic source
manual was derived from a real underwriting document; all supplied policy content
is for this exercise only.

To regenerate chunks after corpus edits:

```bash
python scripts/run_chunking.py --no-caption-api --out data/processed/chunks_v2.json
```

The flag disables vision caption generation; semantic evidence guards can still
call Groq. Bundled captions avoid needing an Anthropic key. Re-chunking can change
evidence IDs: review/remap evaluation anchors before comparing scores. Use the
index lifecycle described in [Architecture](docs/ARCHITECTURE.md) for upgrades
instead of overwriting an active index.

## Tests and evaluation

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false python -m unittest discover -s tests -q
```

Tests use mocks/local stores and do not establish live-model answer quality.
[Evaluation](docs/EVALUATION.md) describes dataset versions, live runners, metrics
and measurement limitations. Live evaluations need provider credentials/services;
inspect each runner's `--help` first.

## Read the design

- [Architecture](docs/ARCHITECTURE.md): paths, state/context, storage and source map.
- [Evaluation](docs/EVALUATION.md): what is measured and how to reproduce checks.
- [Learning journey](docs/LEARNING_JOURNEY.md): decisions and lessons.

`X-Demo-User` defaults to `learner-a` in the browser and is a local identity seam,
not verified authentication. Compose credentials are local demo defaults.
Provider availability, small authored evaluation sets and imperfect LLM judges
remain limitations. No enterprise-scale reliability or security claim is made.
