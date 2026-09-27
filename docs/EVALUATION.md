# Evaluation

This release includes definitions and runners rather than private conversations,
provider caches or every historical run. Generated results belong in the ignored
`results/` directory.

## What the checks establish

| Layer | Measures / checks | Limitation |
| --- | --- | --- |
| Unit contracts | Scope isolation, routes, corrections, guards, commits and index lifecycle | Mocked calls do not measure model accuracy |
| Retrieval | Evidence-group coverage, full coverage, ranking and stage checks | Chunk-ID labels depend on the corpus snapshot |
| Generation | Required-claim recall, groundedness, forbidden claims, relevance, citations, abstention | LLM judges can make systematic mistakes |
| Conversation | Follow-ups, scope changes, reuse and expected actions | Needs multi-turn cases |
| Comparison | Per-scope evidence, citations and completeness | Both sides must be assessed separately |
| Retry | Gate decisions, recovery actions, evidence/answer changes and stopping | A successful retry does not prove it was necessary |
| Operations | Latency, usage/cost estimates and recovery | Cache state and quota affect comparisons |

Group recall is satisfied required evidence groups divided by required groups.
Full coverage asks whether all groups are satisfied for a case. Claim recall
concerns answer content, not how many retrieved chunks are mentioned. Groundedness
checks support for generated claims, including claims outside the required list.

## Data and runners

| Scope | Definitions | Runner |
| --- | --- | --- |
| Original single-manual questions | `eval/*eval_v1.json`, `*eval_v2.json`, split manifests | `scripts/run_eval.py`, `run_generation_eval.py` |
| Graph baseline | Single-manual cases with graph/corpus selected by runner | `scripts/run_graph_dev.py` |
| Conversation | `eval/conversation_eval_v1/` | `scripts/run_conversation_eval.py` |
| Comparison | `eval/comparison_eval_v1/` | `scripts/run_comparison_eval.py` |
| Retry / correction | `eval/retry_eval_v1/` | `scripts/run_retry_eval.py` |
| Excerpt anchors | `eval/excerpts_v1.json`, `excerpts_v2.json` | `scripts/validate_excerpts.py`, `run_chunking_eval.py` |
| Active/candidate indexes | Frozen builds and excerpt anchors | `scripts/run_index_eval.py` |

Inspect `--help` and corpus/version requirements before a live run. Runners have
different defaults; legacy single-manual scores are not automatically comparable
with a four-document graph run. The excerpt-v2 anchors are draft review material,
not independently human-verified ground truth.

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false python -m unittest discover -s tests -q
python scripts/run_conversation_eval.py --help
python scripts/run_comparison_eval.py --help
python scripts/run_retry_eval.py --help
python scripts/run_index_eval.py --help
```

Tests mock provider calls and use temporary/local storage. Public tokenizer/model
assets may be needed on an uncached machine. Live evaluations send questions,
excerpts and answers to configured providers and can incur charges.

## Evidence and interpretation

A prior local release rehearsal used two identical 235-chunk builds. Its ten-case
BM25-only control comparison produced mean group coverage 0.9167 and full coverage
0.80 on both sides. This tests release mechanics, not improved retrieval. Live
Qdrant promotion/rollback and PostgreSQL checkpoint pinning were rehearsed;
changed-content quality approval and in-flight retrieval under load were not
established by that rehearsal. Raw run files remain local.

Report case counts, corpus/build identity, model/prompt versions, cache mode,
latency and costs with quality results. Tune on dev. Public holdout files support
reproduction but must not enter runtime prompts. Tuning on inspected holdout
failures makes those cases unsuitable as an independent test.

Content anchors reduce dependence on sequential chunk IDs. They still need source
review: a matching fragment does not prove table headers or exceptions remain
attached correctly. Compare end-to-end answers as well as retrieved evidence.

Limitations include small authored datasets, vocabulary bias, imperfect judges,
version-dependent labels and provider variability. There is no single current
score representing every capability in this repository.
