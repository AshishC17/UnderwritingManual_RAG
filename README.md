# UnderwritingManual_RAG

RAG system over a synthetic underwriting manual. Qdrant for vector storage, LangGraph for orchestration. Built in two phases:

- **Phase 1 — POC**: chunking, hybrid (BM25 + semantic) retrieval, reranking, a deterministic pipeline, caching, and eval.
- **Phase 2 — Production**: agentic orchestration (conditional nodes, retry logic), eval hardening, semantic drift detection, re-embedding, and scaling to an enterprise-sized user base.

## Status
Phase 1 — auditing source document structure before committing to a chunking strategy.

## Data
`data/raw/` holds the source manual (synthetic, derived from a real underwriting document).
