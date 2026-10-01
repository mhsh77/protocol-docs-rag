# Retrieval experiments (Phase 2)

Same 60-question probe set and hit definition as the chunk-size experiment
(`eval/probe/chunk_probe.jsonl`; a hit = a retrieved chunk contains the verbatim evidence
quote). Chunking: 256 tokens / 32 overlap. Paired bootstrap CIs over the same 60 probes.
Latency measured on the development laptop (4-core mobile CPU, shared with other
workloads), so absolute numbers are pessimistic; the relative differences are what matter.

## Dense vs hybrid vs hybrid + reranker

| Mode | Hit@1 | Hit@5 | MRR@10 | ΔMRR vs dense [95% CI] | Latency p50 / p95 |
|---|---|---|---|---|---|
| Dense (`bge-base-en-v1.5`), the naive baseline | 0.633 | 0.850 | 0.727 | — | 0.18 s / 0.35 s |
| Hybrid: dense + BM25, RRF (k=60) | 0.717 | 0.967 | 0.833 | +0.106 [+0.015, +0.204] | 0.13 s / 0.15 s |
| Hybrid + reranker (`bge-reranker-base`, 20 candidates) | 0.800 | 1.000 | 0.887 | +0.160 [+0.071, +0.253] | 12.76 s / 24.71 s |

Both improvements over dense-only are statistically supported on this probe set.
Caveat: probe questions were written by an LLM looking at the passage, which can share
vocabulary with it and so may favour BM25. The main evaluation uses more varied questions
(multi-section, adversarial, unanswerable) and re-measures this.

## Choosing the reranker

The reranker made the bot too slow on CPU, so I compared smaller cross-encoders and fewer
candidates (input length 384 tokens; chunks are at most ~256 tokens plus heading and query).

| Reranker | Candidates | MRR@10 | Hit@5 | ΔMRR vs bge-reranker-base [95% CI] | Latency p50 / p95 |
|---|---|---|---|---|---|
| `BAAI/bge-reranker-base` (278M) | 20 | 0.887 | 1.000 | — | 12.02 s / 21.39 s |
| `BAAI/bge-reranker-base` | 10 | 0.871 | 0.983 | −0.017 [−0.042, +0.000] | 6.10 s / 8.27 s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | 20 | 0.891 | 0.950 | +0.004 [−0.055, +0.064] | 1.55 s / 2.15 s |
| **`cross-encoder/ms-marco-MiniLM-L-12-v2`** (33M) | 20 | **0.905** | 0.967 | +0.018 [−0.037, +0.075] | 3.53 s / 4.59 s |

Notes:
- Cutting `max_length` from 512 to 384 changed nothing (same MRR, same latency), so cost is
  the model size, not padding.
- **Chosen: MiniLM-L-12, 20 candidates.** Quality is statistically indistinguishable from
  `bge-reranker-base` (best MRR point estimate, two fewer probes in the top 5) at ~3.4×
  lower latency. MiniLM-L-6 is faster still; L-12 was preferred for its higher hit@5.
- Halving candidates for `bge-reranker-base` halves latency but loses a little quality and
  is still slower than either MiniLM model.
