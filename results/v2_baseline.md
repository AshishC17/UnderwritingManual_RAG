# V2 Baseline — Deterministic Hybrid Retrieval

**Date:** 2026-09-03 · **Corpus:** `NovaCred_UW_V0_pdf.pdf` (28 pages, synthetic) · **Eval set:** 27 cases

Supersedes `v1_baseline.md` (commit `a0b1f49`, retrievable from git history).
**The retrieval design is byte-identical to V1** — only the eval set changed, growing from
23 to 27 cases. Scores moved because the test got harder, not because the system changed.
Raw V1 metrics remain at `results/v1_dev.json` / `v1_holdout.json` for comparison.

---

## 1. Design under test

| Stage | Choice | Rationale |
|---|---|---|
| Parser | `pdfplumber` | `pypdf` gives linear text only and cannot recover table structure |
| Heading detection | font size (14/12/11pt = H1/H2/H3, 10pt body) | more reliable than ALL-CAPS matching on this document |
| Table handling | Markdown; whole table if it fits, else row-groups **repeating the header** | a split table without its header is numbers with no column labels |
| Table overlap | **none** | rows are independent records; overlap would duplicate data |
| Cross-page tables | stitched before chunking | 5 tables span page breaks; sibling tables share headers, so continuation needs header-match **and** first-on-page **and** no new caption |
| Diagrams | vision-captioned to text (title + narrative + edge list), cached by image hash | EXHIBIT B had zero extractable text; 2 flowcharts → 3 chunks |
| Chunk size | 512 tokens, 15% overlap, sentence-aware | NVIDIA FinanceBench found 15% optimal at this size for dense financial documents |
| Parent context | **none prepended** | retrieval co-surfaces prose and table chunks anyway |
| Dense embedding | `voyage-context-4`, 1024-dim, cosine | contextualized chunk embeddings. Verified real: identical text alone vs in-group = cosine **0.818** |
| Sparse | FastEmbed `Qdrant/bm25`, IDF server-side | local, free; document = saturated TF, query = term presence |
| Store | Qdrant 1.19.0, one collection, both vectors per point | native hybrid; no separate index to drift |
| Fusion | **RRF** (rank-based) | cosine is bounded, BM25 is not; score fusion would let scale dominate |
| Reranker | Voyage `rerank-2.5`, shortlist 15 | cross-encoder scores query+doc jointly |
| Orchestration | **none** — single deterministic pass | one embed → one search → one rerank. No loops or query rewriting |
| Generation | **none** | no answer produced; retrieval only |

**Corpus:** 87 chunks — 55 primary, 16 table, 13 appendix, 3 flowchart. Median 207 tokens, 17,201 total.

## 2. What changed from V1

Four cases added (23 → 27), all requiring the **Pre-Qual** flowchart (`0033`), which V1 never
exercised — half the diagram-captioning work was previously unverified.

| case | split | difficulty | stress type |
|---|---|---|---|
| M06 | dev | medium | `implicit_stage_flowchart_transition` |
| M07 | holdout | medium | `implicit_stage_flowchart_rework_loop` |
| X08 | dev | extreme | `multi_hop_reused_code_stage_contradiction` |
| X09 | holdout | extreme | `multi_hop_manual_review_stage_contradiction` |

Each phrases the question as *"the stage before full underwriting"* — never "prequalification"
or "prescreen" — so retrieval must infer the stage rather than match its name. Each has a
group only `0033` can satisfy.

**New metric: `stage_accuracy@k`.** Driven by a per-case `evaluation_metrics.stage_accuracy_at_k`
block naming `correct_stage_chunk_ids` and `distractor_stage_chunk_ids`. A case passes when a
correct-stage chunk outranks every distractor. Ranks are reported alongside the verdict, since
"correct at 3, distractor at 1" fails differently from "correct never retrieved".

## 3. Results

### Dev (14 cases — tuned against)

| config | k | grp-recall | full-cov | MRR | prec | stage-acc |
|---|---|---|---|---|---|---|
| dense | 10 | 0.851 | 0.714 | 0.781 | 0.271 | 0.000 |
| sparse | 10 | 0.863 | 0.643 | 0.871 | 0.300 | 0.500 |
| hybrid | 10 | 0.851 | 0.643 | 0.863 | 0.293 | 0.000 |
| **hybrid_rerank** | 10 | **0.929** | **0.857** | **0.940** | 0.336 | 0.500 |

### Holdout (13 cases — never tuned against, run once)

| config | k | grp-recall | full-cov | MRR | prec | stage-acc |
|---|---|---|---|---|---|---|
| dense | 10 | 0.929 | 0.846 | 0.872 | 0.300 | 0.000 |
| sparse | 10 | 0.929 | 0.769 | 0.962 | 0.238 | 0.500 |
| hybrid | 10 | 0.974 | 0.923 | 0.910 | 0.323 | 0.500 |
| **hybrid_rerank** | 10 | **0.974** | **0.923** | **1.000** | 0.315 | **1.000** |

### full-coverage@10 by difficulty

| config | easy | medium | difficult | extreme | | easy | medium | difficult | extreme |
|---|---|---|---|---|---|---|---|---|---|
| | *dev* | | | | | *holdout* | | | |
| dense | 1.00 | 0.75 | 1.00 | 0.25 | | 1.00 | 1.00 | 1.00 | 0.60 |
| sparse | 1.00 | 1.00 | 0.67 | 0.00 | | 1.00 | 1.00 | 1.00 | 0.40 |
| hybrid | 1.00 | 0.75 | 0.67 | 0.25 | | 1.00 | 1.00 | 1.00 | 0.80 |
| hybrid_rerank | 1.00 | 1.00 | 1.00 | 0.50 | | 1.00 | 1.00 | 1.00 | 0.80 |

### Stage disambiguation @10 (correct rank v distractor rank)

| config | dev acc | dev ranks | holdout acc | holdout ranks |
|---|---|---|---|---|
| dense | 0.000 | M06:3v1 X08:7v1 | 0.000 | M07:2v1 X09:3v1 |
| sparse | 0.500 | M06:1v2 X08:−v5 | 0.500 | M07:1v2 X09:2v1 |
| hybrid | 0.000 | M06:2v1 X08:−v1 | 0.500 | M07:1v2 X09:3v1 |
| **hybrid_rerank** | 0.500 | M06:1v2 X08:−v3 | **1.000** | M07:1v2 X09:1v4 |

### Latency (retrieval only; add ~400ms for query embedding)

| config | dev p50 | holdout p50 |
|---|---|---|
| dense | 11 ms | 15 ms |
| sparse | 5 ms | 5 ms |
| hybrid | 8 ms | 9 ms |
| hybrid_rerank | 19 ms | 28 ms |

p95 for `hybrid_rerank` is contaminated by rate-limit backoff and is not reported.

## 3b. Generation

**Generator `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`** — both on Groq, deliberately
different model *families* (OpenAI vs Alibaba), since a model grading its own output has a
self-preference bias that same-family models partly share. `temperature=0` on both so eval
numbers do not wander between runs. Context = the reranked top-10.

| metric | dev (14) | holdout (13) |
|---|---|---|
| claim recall | 0.855 | 0.847 |
| hallucination rate | 0.071 | 0.077 |
| citation validity | 1.000 | 1.000 |
| forbidden assertions | 1 | 2 |

### By difficulty

| difficulty | dev claim-recall | dev halluc | holdout claim-recall | holdout halluc |
|---|---|---|---|---|
| easy | 1.000 | 0.000 | 1.000 | 0.000 |
| medium | 0.817 | 0.000 | 0.583 | 0.333 |
| difficult | 0.800 | 0.000 | 0.867 | 0.000 |
| extreme | 0.825 | 0.250 | 0.931 | 0.000 |

**The one dev hallucination is X08 — the same case that failed retrieval.** Chunk `0033` was
never retrieved (stage metric: `X08:−v3`), so the model answered without the Pre-Qual flow and
asserted *"Resolved evidence at the current stage goes directly to Evidence / Verification"* —
the Full Underwriting behaviour. **It hallucinated the wrong stage because it was handed the
wrong stage.** Retrieval failure cascading into generation failure, traced across two
independent metrics. The fix is at rung 2 of the ladder (retrieval), not in the prompt.

**Citation validity is 1.000 on both splits** — no invented sources. Note the models cite with
CJK brackets `【id】` rather than the ASCII `[id]` the prompt requests; the citation regex
accepts both, and matching only ASCII would have scored every citation invalid.

### JUDGE VALIDATED — kappa 0.900

Hand-checked 2026-09-04 against `qwen/qwen3.8-27b`, the same judge that produced the dev and
holdout numbers above — so this validation applies directly to the table, not to a different
run. 40 verdicts sampled (deliberately balanced present/absent, not random — a random draw is
mostly "absent" and the verdicts worth checking are the "present" ones), with the actual
question and full generated answer reconstructed alongside each claim so the check isn't
limited to trusting the judge's own quoted evidence.

| | result |
|---|---|
| raw agreement | 0.950 (not the headline number — see below) |
| **Cohen's kappa** | **0.900** — target was 0.7–0.8, the human-human ceiling |
| recall (present) | 20/22 = 0.909 |
| recall (absent) | 18/18 = 1.000 — zero false alarms |

Kappa, not raw agreement, is the number that counts: a judge answering "absent" to everything
scores ~0.90 raw agreement while catching zero hallucinations (verified against synthetic
cases before trusting this on real data) — agreement alone can't distinguish a working judge
from a lazy one on an imbalanced label set.

**Two disagreements out of 40, both defensible, both erring toward under- rather than
over-counting:**
- **X03** (required claim, *"Code 120 normally creates a NOAA"*) — the answer ties code 120
  to NOAA via *"the same condition is reflected in the NOAA-120 definition"* rather than
  stating it directly; the judge wanted more explicit phrasing. Only makes claim recall read
  a hair lower than reality.
- **D03** (forbidden claim, *"the application remains open solely because an RFAI also
  fired"*) — the phrase appears literally in a sub-bullet, but the rest of the answer
  contextually overrides it with the real disposition (terminal decline via NOAA, RFAI logged
  only). A genuine edge case in what "asserts" means under later qualification, not a judge
  bug — but the more consequential of the two, since a missed forbidden claim is a missed
  hallucination. Worth a second look if this pattern recurs at scale.

Reproduce: `python scripts/validate_judge.py results/judge_audit_dev.json --sample 40`, fill
`human` in the generated `.labels.json`, then `--score`. Holdout uses the same judge but was
not independently hand-checked — reasonable to trust by extension, not verified directly.

## 4. Headline

**hybrid_rerank holds at 0.923 full-coverage and MRR 1.000 on holdout, with perfect stage
disambiguation.** Adding four hard cases cost dev 0.917 → 0.857 and holdout 1.000 → 0.923 —
the system did not get worse, the test got sharper.

## 5. Findings

**Dense retrieval is systematically stage-blind.** `stage_accuracy = 0.000` on *both* splits,
all four cases: it puts the Full Underwriting flowchart above the Pre-Qual one every time,
even when the question says "before full underwriting". Four cases is small, but the result
is perfectly consistent across two independent splits.

**Reranking is what corrects it** — 0.000 → 0.500 on dev, 0.500 → 1.000 on holdout. Joint
query-document encoding sees the stage cue that a bi-encoder averages away.

**Stage failure is invisible to recall.** hybrid_rerank scores 0.857 full-coverage on dev
while getting the stage wrong half the time. Without `stage_accuracy` this would look like a
healthy configuration.

**X08 fails on every config, and differently from the rest.** The `−` in `X08:−v3` means
chunk `0033` did not appear **within k=10**, while a distractor ranked 3rd. It is in fact
retrieved at **rank 12** — outside the cutoff, not absent from the index. That distinction
matters: the fix is the cutoff, not the retriever. Full walkthrough in §7.

## 6. Caveats

1. **Sample size.** 14 dev / 13 holdout. One case moves full-coverage by ~0.07. Gaps under
   ~0.10 are noise.
2. **Stage accuracy rests on 2 cases per split.** Directionally striking, statistically thin.
   The dense 0.000 result is consistent across 4/4 cases, which is what makes it credible.
3. **Judge validated at kappa 0.900 on dev** (see §3b) — the same judge that scored both
   splits, so dev and holdout generation numbers are trustworthy, not provisional. Holdout was
   not independently hand-checked.
4. **Ground truth keys on positional chunk IDs.** Re-chunking renumbers them and silently
   invalidates labels.
5. **Holdout was run once per version**, aggregates only, never used for tuning.

## 7. Known failure modes

**Compound queries** — dev failures average 5.4 clauses vs 2.7 for passes. X03 proved the
mechanism: one embedding averages all clauses, majority clauses dominate. Same index, same
retriever, only the query changed: full question → chunk `0024` at rank 25; decomposed
sub-question → **rank 1**. Fix is query decomposition, not deeper shortlists.

**Stage confusion** — now measured; see §5.

### Worked example: X08, a retrieval failure that presents as a hallucination

The only dev case that both fails retrieval and hallucinates. Worth tracing in full, because
the surface symptom points at the wrong fix.

**The question.** *"An application is still in the stage before full underwriting when Credit
Freeze returns Code 105. Operations argues that because Code 105 appears later as Credit
Freeze Refresh, resolved evidence should enter Evidence / Verification and resume
full-underwriting checks. Is that correct?"* — Operations is wrong; the model must say so.

**What it needed:** `0011` (Table 3, Prescreen Prequalification Rules), `0012` (reused-code
semantics), `0033` (Pre-Qual flowchart).

**What retrieval gave it (top-10, hybrid_rerank):**

```
 1. 0018  Table 6: Non-Prescreen FULL UNDERWRITING Rules   <- wrong stage
 2. 0039  Table 10: Available RFAIs
 3. 0035  Full Underwriting flowchart                      <- wrong stage
 4. 0034  Full Underwriting flowchart                      <- wrong stage
 6. 0012  needed, present
```

`0011` never retrieved. **`0033` at rank 12 — two places outside the k=10 cutoff.** So the
model received the Full Underwriting rule table and both Full Underwriting flowcharts, and
nothing from the stage the question was about.

**What it did with that.** It understood the question — the answer opens *"At the
pre-qualification point..."* — but had to cite **Table 6** (Full Underwriting) to support it,
because Table 3 was absent. It then got the reused-code reasoning right from `0012`, and got
the final question right (*"Does any of these steps grant final approval? – No."*). But for the
route it wrote:

> *"Rule Result? – RFAI → the flowchart sends the application to **Evidence / Verification**
> [0035][0034] ... loops back to **Run Full Underwriting Policy & Credit Checks**"*

That is the forbidden claim nearly verbatim. **`Evidence / Verification` exists only in the
Full Underwriting flowchart.** The Pre-Qual equivalent is
`Need More Information (NMI) --info received--> Run Prescreen Prequalification Checks` — a
different node with a different destination. The model described the Full Underwriting rework
loop because that is the only rework loop it was shown.

**The lesson.** The model reasoned correctly over the wrong evidence. Reading "asserted a
forbidden claim" and going to work on the prompt would have been wasted effort — no prompt
change puts a missing chunk into context. It took two metrics together to localise it: stage
accuracy showing `X08:−v3`, then hallucination on the same case.

**The candidate fix is Tier-0** — `0033` is at rank 12, so raising `k` from 10 to 15 puts it in
context. But **one case at rank 12 is not evidence that k=10 is wrong in general**: raising `k`
adds tokens, latency, and distraction across all 27 cases, and "lost in the middle" is a real
effect. This is a change to make against the full eval and measure, not a one-case patch.

**Abbreviation mismatch — still untested by the eval.** The corpus uses `NOAA` 97 times and
never expands it; `RFAI` 64 times. Measured BM25 top-5 overlap between document vocabulary
and plausible user phrasing: NOAA vs "Notice of Adverse Action" = **0/5**, RFAI vs "Request
for Additional Information" = 1/5, "Intraday Account Review" vs IAR = 1/5. Hybrid does not
help — the typed token is absent from the corpus. All 27 cases use the document's own
vocabulary, so this remains invisible to the eval.

## 8. Next

1. ~~Validate the judge~~ — **done, kappa 0.900** (§3b).
2. **Sweep `k`** against the full eval — X08 needs 12, but raising `k` costs tokens, latency
   and distraction on all 27 cases. Measure, do not patch the one case.
3. Metadata filtering for stage, with `stage_accuracy` as the instrument that proves it worked.
4. Query decomposition (LangGraph) — addresses the compound-query failure class, evidence-backed
   on X03 (full question → chunk `0024` rank 25; sub-question → rank 1).
5. Add abbreviation cases to ground truth **before** attempting HyDE — HyDE is a hypothesis with
   zero eval cases testing it, unlike decomposition.
6. HyDE — targets the abbreviation gap.
7. Fix `rule_codes_referenced` regex — `\b(1[0-4][0-9])\b` matches `$100` and "120 days".

## 9. Reproducing

```bash
docker compose up -d
python scripts/run_chunking.py
python scripts/run_embed.py
python scripts/run_eval.py --split dev
python scripts/run_generation_eval.py --split dev --export-judge results/judge_audit_dev.json
python scripts/validate_judge.py results/judge_audit_dev.json --sample 40
```

Requires `VOYAGE_API_KEY` (retrieval) and `GROQ_API_KEY` (generation) in `.env`.

`--split dev_v1` / `holdout_v1` re-run the original 23-case set.
Raw metrics: `results/v2_dev.json`, `results/v2_holdout.json`.
Holdout prints aggregates only by policy — tune on dev, never on holdout.
