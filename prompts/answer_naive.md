# Answer prompt — naive baseline

Used only by the `naive_rag` eval configuration, to measure what the grounding rules in
`answer_v2.md` buy. It is the generic "use the context to answer" prompt found in most RAG
tutorials: no instruction to avoid outside knowledge, no citation rules, no false-premise
rule. It keeps the same JSON output shape so the same pipeline and metrics apply, and it does
allow declining, so abstention is not artificially forced to zero.

## System

You are a helpful assistant for developers using {protocol_name}.

## User

Use the following context to answer the question.

Context:
{sources}

Question: {question}

Return JSON with:
- "answer": your answer
- "abstain": true if you cannot answer the question, otherwise false
- "closest_source": null

Each source block is rendered as in answer_v2.md.
