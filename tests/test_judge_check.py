from __future__ import annotations

from pathlib import Path

import pytest

from docrag.eval.judge import Verdict
from docrag.eval.judge_check import agreement, render_sheet, sample_for_review
from tests.test_report import rec


def judged(qid: str, grade: int):  # type: ignore[no-untyped-def]
    return rec(qid, verdict=Verdict(correctness=grade))


RECS = [judged(f"q{i}", i % 3) for i in range(12)]


def test_sample_is_stratified_over_grades() -> None:
    picked = sample_for_review(RECS, n=6)
    grades = sorted(r.verdict.correctness for r in picked if r.verdict)
    assert grades == [0, 0, 1, 1, 2, 2]


def test_sheet_hides_judge_grade_and_round_trips(tmp_path: Path) -> None:
    picked = sample_for_review(RECS, n=3)
    sheet = render_sheet(picked)
    assert "correctness" not in sheet.lower().replace("correctness_reason", "")
    # human agrees on two items, disagrees on one, skips none
    filled_lines = []
    answers = iter(
        [
            str(picked[0].verdict.correctness),
            str(picked[1].verdict.correctness),
            "0" if picked[2].verdict.correctness else "2",
        ]  # type: ignore[union-attr]
    )
    for line in sheet.splitlines():
        filled_lines.append(line + next(answers) if line.startswith("Your grade") else line)
    path = tmp_path / "sheet.md"
    path.write_text("\n".join(filled_lines), encoding="utf-8")
    res = agreement(path, picked)
    assert res.n == 3
    assert res.agreement == pytest.approx(2 / 3)
    assert len(res.disagreements) == 1


def test_empty_grades_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "sheet.md"
    path.write_text(render_sheet(RECS[:2]), encoding="utf-8")
    with pytest.raises(ValueError, match="no graded items"):
        agreement(path, RECS[:2])
