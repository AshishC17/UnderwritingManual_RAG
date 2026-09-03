# V1 Baseline — Deterministic Hybrid Retrieval

**Date:** 2026-09-03 · **Commit:** `bd44264` · **Corpus:** `NovaCred_UW_V0_pdf.pdf` (28 pages, synthetic)

The first end-to-end retrieval pipeline, measured on both eval splits. This file records
*what the design was* alongside *what it scored*, so later iterations can be compared
against a known configuration rather than a remembered one.

---

## 1. Design under test

| Stage | Choice | Rationale |
|---|---|---|
| Parser | `pdfplumber` | `pypdf` gives linear text only and cannot recover table row/column structure |
| Heading detection | font size (14pt=H1, 12pt=H2, 11pt=H3, 10pt=body) | more reliable than ALL-CAPS pattern matching on this document |
| Table handling | rendered to Markdown; whole table if it fits, else row-groups **repeating the header** | a split table without its header is numbers with no column labels |
| Table overlap | **none** | rows are independent records; overlap would duplicate data without aiding retrieval |
| Cross-page tables | stitched before chunking | 5 tables span page breaks; sibling tables share headers, so continuation requires header-match **and** first-on-page **and** no new caption |
| Diagrams | vision-captioned to text (title + narrative + explicit edge list), cached by image hash | EXHIBIT B had zero extractable text; 2 flowcharts, 3 chunks |
| Chunk size | 512 tokens, 15% overlap, sentence-aware | NVIDIA FinanceBench found 15% optimal at this size for dense financial documents |
| Parent context | **none prepended** | measured decision: retrieval co-surfaces prose and table chunks anyway; prepending would duplicate tokens |
| Dense embedding | `voyage-context-4`, 1024-dim, cosine | contextualized chunk embeddings — gives the parent-context benefit without token duplication. Verified real: identical text embedded alone vs in-group gives cosine **0.818** |
| Sparse | FastEmbed `Qdrant/bm25`, IDF computed server-side | local, free; document side = saturated TF, query side = term presence |
| Store | Qdrant 1.19.0, one collection, both vectors per point | native hybrid; no separate BM25 index to drift |
| Fusion | **RRF** (rank-based) | cosine is bounded, BM25 is not; score-level fusion would let scale rather than relevance dominate |
| Reranker | Voyage `rerank-2.5`, shortlist 15 | cross-encoder scores query+doc jointly; bi-encoders cannot judge interaction |
| Orchestration | **none** — single deterministic pass | one embed → one search → one rerank. No loops, retries, or query rewriting |
| Generation | **none** | no answer is produced; this measures retrieval only |

**Corpus:** 87 chunks — 55 primary, 16 table, 13 appendix, 3 flowchart. Median 207 tokens,
max 495, 17,201 total.

## 2. Metrics

Scored against `evidence_groups` (AND across groups, OR within each), never
`supporting_chunk_ids` — the latter is related enrichment and is often disjoint from the
evidence actually needed.

- **group recall@k** — fraction of required evidence groups satisfied
- **full-coverage@k** — every group satisfied (the answerability bar)
- **MRR** — rank of the first relevant chunk
- **precision@k** — logged, low weight: 1–3 relevant chunks in 87 caps it near 3/k

## 3. Results

### Dev (12 cases — tuned against)

| config | k | grp-recall | full-cov | MRR | prec |
|---|---|---|---|---|---|
| dense | 10 | 0.882 | 0.750 | 0.872 | 0.300 |
| sparse | 10 | 0.896 | 0.667 | 0.892 | 0.333 |
| hybrid | 10 | 0.882 | 0.667 | 0.944 | 0.325 |
| **hybrid_rerank** | 10 | **0.972** | **0.917** | **1.000** | 0.375 |

### Holdout (11 cases — never tuned against, run once)

| config | k | grp-recall | full-cov | MRR | prec |
|---|---|---|---|---|---|
| dense | 10 | 0.977 | 0.909 | 0.955 | 0.336 |
| sparse | 10 | 0.947 | 0.818 | 1.000 | 0.255 |
| **hybrid** | 10 | **1.000** | **1.000** | 0.955 | 0.345 |
| **hybrid_rerank** | 10 | **1.000** | **1.000** | **1.000** | 0.355 |

### full-coverage@10 by difficulty

| config | easy | medium | difficult | extreme | | easy | medium | difficult | extreme |
|---|---|---|---|---|---|---|---|---|---|
| | *dev* | | | | | *holdout* | | | |
| dense | 1.00 | 0.67 | 1.00 | 0.33 | | 1.00 | 1.00 | 1.00 | 0.75 |
| sparse | 1.00 | 1.00 | 0.67 | 0.00 | | 1.00 | 1.00 | 1.00 | 0.50 |
| hybrid | 1.00 | 0.67 | 0.67 | 0.33 | | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid_rerank | 1.00 | 1.00 | 1.00 | 0.67 | | 1.00 | 1.00 | 1.00 | 1.00 |

### Latency (retrieval only; add ~400ms for query embedding)

| config | p50 ms | p95 ms |
|---|---|---|
| dense | 19 | 223 |
| sparse | 5 | 17 |
| hybrid | 9 | 61 |
| hybrid_rerank | 471 | *contaminated — see caveats* |

## 4. Headline

**V1 reaches full coverage at k=10 on the holdout split with hybrid retrieval.** Gains
measured on dev transferred; no overfitting detected.

## 5. Caveats — read before trusting any number

1. **Sample size.** 12 dev / 11 holdout. One case moves full-coverage by ~0.09. Gaps under
   ~0.10 are noise. The dense/sparse/hybrid ordering is **not** significant.
2. **Holdout saturates at k=10.** Both hybrid and hybrid_rerank hit 1.000, so holdout does
   **not** discriminate between them. The reranker's value rests on dev (0.667 → 0.917),
   where it was decisive.
3. **At k=5 on holdout, hybrid (0.818) beat hybrid_rerank (0.636)** — the reverse of dev.
   Two cases at this sample size; do not read a trend into it.
4. **`hybrid_rerank` p95 latency is a measurement artifact.** It includes rate-limit backoff
   sleeps, not model time. p50 (471ms) is real; p95 is not.
5. **Retrieval only.** No generation, so `required_claims` / `forbidden_claims` are unused
   and answer quality is entirely unmeasured.
6. **Ground truth keys on positional chunk IDs.** Any re-chunking renumbers them and
   silently invalidates the labels. Re-label or move to content hashes before changing
   chunking.

## 6. Known failures and diagnoses

**X03** (`exception_reference_and_non_final_pass`) — failed on every config in dev.
Diagnosed to a **compound-query failure**, not chunking or retrieval:

- The question asks four things at once. One embedding averages all four clauses; the three
  override/approval clauses dominate and drown out the audit-trail clause.
- Proof, same index and retriever, only the query changed:
  `full compound question → chunk 0024 at rank 25`; `decomposed sub-question → rank 1`.
- Contributing factor: chunk `0024` is mixed-topic (Intraday Account Review + Fraud
  Consortium), so its embedding is also diluted. Secondary — a diluted chunk could not rank
  #1 for any query.
- **Correct fix is query decomposition (Tier-1), not raising shortlist depth (Tier-0).**

**Compound queries generalise as a failure mode.** Clause count vs outcome across dev:
failed cases average **5.4 clauses**, passed cases **2.7**. Every question with ≤3 clauses
passed. (n=12, crude clause counter — a strong hint, not proof.)

**Abbreviation mismatch — untested by the current eval, and severe.** The corpus uses `NOAA`
97 times and **never expands it**; `RFAI` 64 times with only an incidental near-phrase.
Measured BM25 top-5 overlap between how the document writes a term and how a user might ask:

| doc says | user asks | overlap |
|---|---|---|
| `NOAA` | `Notice of Adverse Action` | **0/5** |
| `RFAI` | `Request for Additional Information` | 1/5 |
| `Intraday Account Review` | `IAR` | 1/5 |
| `Fraud Consortium` | `FC` | 2/5 |

Hybrid retrieval does not help here — the token the user typed is absent from the corpus, so
BM25 is actively the wrong tool and the burden falls entirely on dense retrieval knowing the
expansion. **All 23 ground-truth cases use the document's own vocabulary, so this failure
mode is currently invisible to the eval.**

## 7. Next iteration

1. Add abbreviation cases to ground truth so V2's improvement is measurable, not assumed.
2. Query decomposition — addresses 4 of 5 dev failures, not just X03.
3. HyDE — directly targets the abbreviation gap (generate a hypothetical answer in document
   vocabulary, retrieve with that).
4. Move orchestration to LangGraph; decomposition and HyDE both mean multiple retrievals per
   question, which crosses out of V1's deterministic single-pass design.
5. Fix the `rule_codes_referenced` regex — `\b(1[0-4][0-9])\b` matches any integer 100–149
   and picks up `$100` and "120 days" as codes.

## 8. Reproducing

```bash
docker compose up -d
python scripts/run_chunking.py
python scripts/run_embed.py
python scripts/run_eval.py --split dev
```

Raw metrics: `results/v1_dev.json`, `results/v1_holdout.json`.
Holdout prints aggregates only by policy — tune on dev, never on holdout.
