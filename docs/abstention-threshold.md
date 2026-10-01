# Abstention threshold (retrieval-score gate)

Tuned on the **dev split only** (29 questions, run `run-4a85a84265`): abstain without
calling the LLM when the top reranker score is below the threshold. Objective: maximise
correct abstention minus false abstention, with false abstention <= 10%.

| Threshold | Correct abstention | False abstention |
|---|---|---|
| none (model only) | 100.0% | 0.0% |
| 2.847 | 100.0% | 5.0% |
| 2.996 | 100.0% | 10.0% |
| 3.424 | 100.0% | 15.0% |
| 4.276 | 100.0% | 20.0% |
| 5.018 | 100.0% | 25.0% |
| 5.061 | 100.0% | 30.0% |
| 5.732 | 100.0% | 35.0% |
| 5.744 | 100.0% | 40.0% |
| 5.772 | 100.0% | 45.0% |
| 5.892 | 100.0% | 50.0% |
| 6.036 | 100.0% | 55.0% |
| 6.156 | 100.0% | 60.0% |
| 6.655 | 100.0% | 65.0% |
| 6.848 | 100.0% | 70.0% |
| 7.292 | 100.0% | 75.0% |
| 7.499 | 100.0% | 80.0% |
| 7.803 | 100.0% | 85.0% |
| 8.039 | 100.0% | 90.0% |
| 8.569 | 100.0% | 95.0% |
| 10.235 | 100.0% | 100.0% |

**Chosen: none** (correct abstention 100.0%, false abstention 0.0% on dev).

Set it as `generation.min_rerank_score` in `config/uniswap.yaml`. With few dev
questions this is a coarse estimate; the test split measures it out of sample.

## Notes

- On dev, the model's own abstention already declines all 9 should-decline questions with
  no false abstentions, so the gate cannot improve the objective and is left off (ties go to
  the simpler system).
- The top reranker scores separate cleanly on dev: every should-decline question scored
  <= 2.44 and every answerable one >= 2.85. A gate in that gap would cost nothing on dev and
  would save an LLM call on clearly off-topic questions, but 29 questions are too few to
  trust a threshold. Whether this separation holds is checked on the test split as a
  clearly labelled secondary analysis (simulated from the test records, no extra calls),
  not by tuning on test.
- This run is unjudged (`--no-judge`): threshold tuning needs only retrieval scores and the
  abstain decision, which leaves the judge's daily token quota for the test run.
- **Pre-registered for the secondary test analysis** (written before any test-split result
  existed): simulated gate at **2.65**, the midpoint of the dev gap between 2.44 and 2.85.
  Reproduce with `rag gate-analysis <test-run-dir> --threshold 2.65`.
