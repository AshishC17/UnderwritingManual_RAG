# Learning journey

This repository records a progression from document search to a stateful
assistant. It is self-learning work, not a claim of operating a deployed
enterprise underwriting service.

| Stage | Question explored | Lesson retained |
| --- | --- | --- |
| Ingestion | How do headings, tables, footnotes and diagrams become evidence? | Chunk boundaries and structure affect answerability; diagrams here become text captions |
| Hybrid retrieval | Why combine semantic and lexical search? | Codes and paraphrases need different signals; rerankers cannot recover unseen candidates |
| Evaluation | Does a plausible answer have complete support? | Evidence coverage, completeness and groundedness measure different failures |
| Conversation | What does “the other lender” mean? | State differs from LLM context; conversation questions need conversation evidence |
| Comparison | Can one answer keep lenders separate? | Retrieve and attribute evidence per lender/version |
| Correction | Revise, search again, clarify or stop? | Model judgments need bounded actions and measured outcomes |
| Persistence | What survives restart or duplicate requests? | Checkpoints, transcripts and execution records have different responsibilities |
| Index releases | How can corpus updates be tested and reversed? | Freeze builds, compare candidates, promote aliases and pin turn provenance |

The main architecture is a controlled workflow with model-driven interpretation
and evaluation. Fixed nodes and retry limits do not alone determine whether a
system is called agentic. Explain the actual model decisions, permitted actions
and measured trade-offs.

Warehouse Q&A and richer tool selection are separate work. The semantic model,
compiler and synthetic warehouse are outside this document-RAG release.
Application containerization is also a later step; only local database services
are containerized in this snapshot.
