# Chunk size experiment

**Question:** what chunk size / overlap should the pipeline use, and does merging small
sections help?

**Setup**
- Probe set: 60 questions in `eval/probe/chunk_probe.jsonl`, drafted by an LLM from a
  seeded random section of 60 different docs, each with a verbatim evidence quote that was
  checked to be an exact substring of that section. This set is separate from the final
  eval set, so the chunking choice is not tuned on the questions it is scored on.
- Retrieval: dense only (`BAAI/bge-base-en-v1.5`), so the chunking effect is measured
  without BM25 or reranking on top.
- Hit = a retrieved chunk contains the evidence quote. Labels are quotes, not chunk ids,
  so every chunking config is scored against the same labels.
- `hit@2000tok` = hit within the first 2,000 tokens of ranked context. Bigger chunks win
  plain hit@k simply by containing more text; this metric compares them at equal cost.
- `top5_tokens` = mean tokens in the top-5 chunks, i.e. what the generator reads per question.
- Raw per-probe results: `chunk_size_raw.json`. Regenerate with the script in the repo history
  (`docrag.experiments.chunk_size.run_grid`).

## Results

| max_tokens | overlap | merge<N | hit@1 | hit@5 | mrr@10 | hit@2000tok | n_chunks | median_tokens | top5_tokens |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 0 | 0 | 0.583 | 0.850 | 0.696 | 0.950 | 5045 | 101 | 498 |
| 128 | 16 | 0 | 0.600 | 0.833 | 0.699 | 0.950 | 5059 | 101 | 501 |
| 256 | 0 | 0 | 0.617 | 0.850 | 0.710 | 0.933 | 3553 | 146 | 864 |
| **256** | **32** | **0** | **0.633** | **0.850** | **0.727** | **0.967** | 3562 | 146 | **857** |
| 512 | 0 | 0 | 0.583 | 0.850 | 0.714 | 0.933 | 2676 | 136 | 1223 |
| 512 | 64 | 0 | 0.567 | 0.850 | 0.705 | 0.917 | 2680 | 137 | 1223 |
| 1024 | 0 | 0 | 0.550 | 0.817 | 0.683 | 0.883 | 2366 | 122 | 1374 |
| 1024 | 128 | 0 | 0.550 | 0.833 | 0.686 | 0.900 | 2366 | 122 | 1378 |
| 256 | 32 | 64 | 0.633 | 0.850 | 0.728 | 0.950 | 3148 | 170 | 911 |
| 512 | 64 | 64 | 0.567 | 0.917 | 0.706 | 0.917 | 2159 | 207 | 1342 |

MRR difference against the previous default (512/64, no merging), paired bootstrap over
the same 60 probes, 2,000 resamples:

| config (max/overlap/merge) | MRR@10 | ΔMRR | 95% CI |
|---|---|---|---|
| 128/0/0 | 0.696 | -0.009 | [-0.113, +0.096] |
| 128/16/0 | 0.699 | -0.005 | [-0.110, +0.096] |
| 256/0/0 | 0.710 | +0.006 | [-0.066, +0.076] |
| 256/32/0 | 0.727 | +0.022 | [-0.042, +0.087] |
| 512/0/0 | 0.714 | +0.009 | [+0.000, +0.026] |
| 1024/0/0 | 0.683 | -0.021 | [-0.082, +0.031] |
| 1024/128/0 | 0.686 | -0.018 | [-0.080, +0.034] |
| 256/32/64 | 0.728 | +0.023 | [-0.041, +0.088] |
| 512/64/64 | 0.706 | +0.001 | [-0.009, +0.011] |

## Conclusion

- **Retrieval quality does not separate the configs.** With 60 probes every MRR confidence
  interval includes zero. Point estimates favour 256/32 and put 1024 last, but this is
  not a statistically supported ranking, and I am not claiming one.
- **Cost does separate them.** 256-token chunks send ~860 tokens of context per question
  against ~1,220 for 512, about 30% fewer prompt tokens, with retrieval that is at least
  as good on every metric above. On a free tier capped at 200K tokens/day per model, that
  is directly more evaluated questions and more bot answers per day.
- **128 is cheaper still**, but its point estimates are lower and 100-token chunks often
  cut a sentence from the context it needs. The probe metric only checks that the quote is
  inside the chunk, not that the chunk carries enough context to answer, so I did not go
  below 256.
- **Merging small sections did not help** measurably (256/32 vs 256/32/64: +0.001 MRR,
  one probe worse on hit@2000tok, more tokens). It stays in the code, tested but
  disabled (`min_section_tokens: 0`), and is listed under "what did not work".

**Chosen default: `max_tokens=256`, `overlap_tokens=32`, `min_section_tokens=0`.**

## Limitations

- 60 probes, single-sentence evidence; multi-section questions are not represented here
  (they are in the main eval).
- Probe questions were written by an LLM looking at the passage, so they may share
  vocabulary with it more than real user questions do.
- Dense retrieval only; the hybrid/reranker effect is measured in the main eval.
