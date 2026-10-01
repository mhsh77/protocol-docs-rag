from __future__ import annotations

from typing import Any

import pytest

from docrag.eval.threshold import best_threshold, simulate
from tests.test_report import rec


def r(qid: str, expected: str, score: float, abstained: bool = False, **kw: Any):  # type: ignore[no-untyped-def]
    return rec(
        qid, expected=expected, top_rerank_score=score, abstained=abstained, split="dev", **kw
    )


RECORDS = [
    r("u1", "abstain", -5.0),  # unanswerable, low score, model answered anyway
    r("u2", "abstain", -4.0),
    r("u3", "abstain", 2.0, abstained=True),  # model abstained by itself
    r("a1", "answer", 3.0),
    r("a2", "answer", 4.0),
    r("a3", "answer", -4.5),  # answerable but low score: a gate would wrongly abstain
]


def test_simulate_combines_gate_and_model_abstention() -> None:
    none = simulate(RECORDS, None)
    assert none.correct_abstention == pytest.approx(1 / 3)
    assert none.false_abstention == 0.0
    gated = simulate(RECORDS, -3.0)
    assert gated.correct_abstention == 1.0
    assert gated.false_abstention == pytest.approx(1 / 3)


def test_best_threshold_respects_false_abstention_cap() -> None:
    best = best_threshold(RECORDS, max_false_abstention=0.0)
    assert best.false_abstention == 0.0
    # Gating u2 (-4.0) would also gate answerable a3 (-4.5), so only u1 (-5.0) is caught.
    assert best.correct_abstention == pytest.approx(2 / 3)
    assert best.threshold is not None and -5.0 < best.threshold <= -4.5


def test_refuses_test_split() -> None:
    with pytest.raises(ValueError):
        best_threshold([rec("x", split="test")])
