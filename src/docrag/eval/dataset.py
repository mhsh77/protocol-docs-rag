"""Eval question set schema and gold-label resolution.

Gold labels are verbatim evidence quotes, not chunk ids, so the same question set stays
valid when chunking changes. At eval time each quote resolves to the chunks containing it.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from docrag.ingest.chunking import Chunk


class Category(StrEnum):
    FACTUAL = "factual"
    MULTI_SECTION = "multi_section"
    NUMERIC = "numeric"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class Expected(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    CORRECT_PREMISE = "correct_premise"  # answer, but explicitly correct the false premise


class Evidence(BaseModel):
    doc_id: str
    quote: str


class EvalQuestion(BaseModel):
    id: str
    split: str  # "dev" | "test"
    category: Category
    subtype: str | None = None  # adversarial: false_premise | outdated_term | other_protocol
    question: str
    expected: Expected
    reference_answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    source: str = "llm_draft"  # llm_draft | manual
    reviewed: bool = False
    notes: str | None = None

    @property
    def answerable(self) -> bool:
        return self.expected is not Expected.ABSTAIN


_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """Whitespace- and inline-markup-insensitive form used for evidence matching."""
    s = s.replace("**", "").replace("`", "")
    return _WS.sub(" ", s).strip().lower()


def resolve_evidence(q: EvalQuestion, chunks: list[Chunk]) -> list[set[str]]:
    """For each evidence quote, the set of chunk ids whose text contains it."""
    normed = [(c.chunk_id, c.doc_id, norm(c.text)) for c in chunks]
    groups = []
    for ev in q.evidence:
        quote = norm(ev.quote)
        groups.append({cid for cid, doc, text in normed if doc == ev.doc_id and quote in text})
    return groups


def read_questions(path: Path) -> list[EvalQuestion]:
    with open(path, encoding="utf-8") as f:
        return [EvalQuestion.model_validate(json.loads(line)) for line in f if line.strip()]


def write_questions(questions: list[EvalQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(q.model_dump_json(exclude_none=True) + "\n")


def append_question(q: EvalQuestion, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(q.model_dump_json(exclude_none=True) + "\n")
