"""Human spot check of LLM-judge decisions.

`render_sheet` samples judged records (stratified over the judge's correctness grades, so
disagreements on all three grades are visible) and writes a Markdown sheet that needs no
domain knowledge: the reviewer compares the assistant's answer with the reference answer
and writes their own grade. `agreement` parses the filled sheet and reports raw agreement
and Cohen's kappa against the judge.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from pydantic import BaseModel

from docrag.eval.metrics import cohens_kappa
from docrag.eval.runner import Record

_ITEM_RE = re.compile(r"^### (\S+) / (\S+)$", re.M)
_GRADE_RE = re.compile(r"^Your grade \(2/1/0\):[ \t]*([012])?[ \t]*$", re.M)


def sample_for_review(records: list[Record], n: int = 20, seed: int = 5) -> list[Record]:
    judged = [r for r in records if r.verdict is not None]
    rng = random.Random(seed)
    by_grade: dict[int, list[Record]] = {0: [], 1: [], 2: []}
    for r in judged:
        assert r.verdict is not None
        by_grade[r.verdict.correctness].append(r)
    for bucket in by_grade.values():
        rng.shuffle(bucket)
    picked: list[Record] = []
    while len(picked) < min(n, len(judged)):  # round-robin over grades
        for grade in (0, 1, 2):
            if by_grade[grade] and len(picked) < n:
                picked.append(by_grade[grade].pop())
    rng.shuffle(picked)
    return picked


def render_sheet(records: list[Record]) -> str:
    lines = [
        "# Judge spot check",
        "",
        "No Uniswap knowledge needed. For each item, compare the **assistant's answer** with the",
        "**reference answer** and write a grade on the `Your grade` line:",
        "",
        "- **2** = the answer says what the reference says (extra correct detail is fine)",
        "- **1** = partly: some of the reference is there, but something important is missing",
        "- **0** = wrong, misleading, or misses the point",
        "",
        "For questions marked *false premise*, the answer must say the question's assumption",
        "is wrong. Leave the line empty to skip an item. The judge's grade is hidden on purpose.",
        "",
    ]
    for r in records:
        kind = r.category + (", false premise" if r.expected == "correct_premise" else "")
        lines += [
            f"### {r.config} / {r.qid}",
            "",
            f"**Question** ({kind}): {r.question}",
            "",
            f"**Reference answer:** {r.reference or '(none)'}",
            "",
            f"**Assistant's answer:** {r.answer}",
            "",
            "Your grade (2/1/0): ",
            "",
        ]
    return "\n".join(lines)


class Agreement(BaseModel):
    n: int
    agreement: float
    kappa: float
    disagreements: list[tuple[str, int, int]]  # (item, human, judge)


def agreement(sheet: Path, records: list[Record]) -> Agreement:
    text = sheet.read_text(encoding="utf-8")
    items = _ITEM_RE.findall(text)
    grades = _GRADE_RE.findall(text)
    if len(items) != len(grades):
        raise ValueError(f"sheet has {len(items)} items but {len(grades)} grade lines")
    by_key = {(r.config, r.qid): r for r in records}
    human: list[str] = []
    judge: list[str] = []
    disagreements = []
    for (config, qid), g in zip(items, grades, strict=True):
        rec = by_key.get((config, qid))
        if not g or rec is None or rec.verdict is None:
            continue
        human.append(g)
        judge.append(str(rec.verdict.correctness))
        if g != judge[-1]:
            disagreements.append((f"{config}/{qid}", int(g), rec.verdict.correctness))
    if not human:
        raise ValueError("no graded items in the sheet")
    agree = sum(h == j for h, j in zip(human, judge, strict=True)) / len(human)
    return Agreement(
        n=len(human),
        agreement=agree,
        kappa=cohens_kappa(human, judge),
        disagreements=disagreements,
    )
