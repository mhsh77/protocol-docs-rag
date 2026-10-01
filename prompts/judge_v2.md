# Judge prompt — v2

Used by `docrag.eval.judge`. Temperature 0. One call per (question, answer) grades
correctness and faithfulness together, to stay inside free-tier token budgets.

Changes from v1 (from a 3-question smoke run, before any reported eval run):
- v1 marked a correct answer "partial" because it omitted a detail the reference happened
  to include but the question did not ask for. v2 defines key facts as what the question
  asks for; extra reference details are optional.
- v1 told the judge references were "written by a human". They were LLM-drafted and
  reviewed against the source text, so v2 says "checked against the documentation".

Design notes
- **Correctness** is graded against the reviewed reference answer, on a 3-point
  rubric (0/1/2) with written anchors, so "partially correct" is defined, not vibes.
- **Faithfulness** is graded against the *retrieved context the assistant actually saw*,
  not against the reference: an answer can be correct but unfaithful (used outside
  knowledge) or faithful but incorrect (retrieval surfaced the wrong section).
- The judge first splits the answer into atomic claims and labels each one. The
  hallucination rate is computed in code from these labels, not by asking for a score.
- The judge never sees which retrieval configuration produced the answer.

## System

You are a strict evaluator of a documentation question-answering assistant. You grade
precisely, follow the rubric literally, and do not reward fluent but unsupported text.

## User

Question:
{question}

Expected behaviour: {expected_behavior}

Reference answer (checked against the documentation):
{reference}

Retrieved context the assistant was given:
<context>
{context}
</context>

Assistant's answer:
<answer>
{answer}
</answer>

Grade the answer.

1. CLAIMS. Split the assistant's answer into atomic factual claims (one fact each; ignore
   citation markers, greetings and statements like "the documentation does not say").
   For each claim decide "supported": true only if the retrieved context states or directly
   implies it. Background knowledge does not count, even if the claim is true.

2. CORRECTNESS against the reference answer and expected behaviour. "Key facts" are the
   facts needed to answer what the question actually asks. Extra details in the reference
   that the question does not ask for are optional: do not lower the grade for omitting
   them, and do not lower it for correct extra detail in the answer.
   - 2 = correct: contains the key facts of the reference, no contradicting statements.
     For expected behaviour "correct_premise", the false premise is explicitly corrected.
   - 1 = partially correct: some key facts present but important parts missing, or a
     minor inaccuracy that would not mislead a developer.
   - 0 = incorrect: wrong, misleading, missing the key fact, or accepts a false premise.
   If the assistant declined to answer, set correctness to 0 and explain.

Return JSON:
{{
  "claims": [{{"claim": "...", "supported": true}}],
  "correctness": 0,
  "correctness_reason": "one or two sentences"
}}
