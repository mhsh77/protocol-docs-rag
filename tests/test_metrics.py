import math

import pytest

from docrag.eval.metrics import (
    bootstrap_ci,
    cohens_kappa,
    evidence_recall_at_k,
    first_hit_rank,
    hit_at_k,
    paired_bootstrap_diff,
    percentile,
    reciprocal_rank,
)

RET = ["a", "b", "c", "d", "e"]


def test_rank_hit_and_mrr() -> None:
    assert first_hit_rank(RET, {"c", "e"}) == 3
    assert hit_at_k(RET, {"c"}, 3) == 1.0
    assert hit_at_k(RET, {"c"}, 2) == 0.0
    assert reciprocal_rank(RET, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RET, {"z"}) == 0.0
    assert reciprocal_rank(RET, {"e"}, cutoff=4) == 0.0


def test_evidence_recall_requires_each_piece() -> None:
    groups = [{"a", "x"}, {"d"}, {"z"}]  # piece 1 in two chunks, piece 3 not retrieved
    assert evidence_recall_at_k(RET, groups, 5) == pytest.approx(2 / 3)
    assert evidence_recall_at_k(RET, groups, 1) == pytest.approx(1 / 3)
    assert evidence_recall_at_k(RET, [], 5) == 0.0


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert math.isnan(percentile([], 50))


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    vals = [1.0] * 30 + [0.0] * 10
    lo, hi = bootstrap_ci(vals)
    assert lo <= 0.75 <= hi
    assert bootstrap_ci(vals) == (lo, hi)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_paired_diff_detects_consistent_improvement() -> None:
    a = [0.0] * 20 + [1.0] * 20
    b = [1.0] * 40
    d, lo, _hi = paired_bootstrap_diff(a, b)
    assert d == pytest.approx(0.5)
    assert lo > 0  # CI excludes zero
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_cohens_kappa_known_values() -> None:
    assert cohens_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == pytest.approx(1.0)
    # 50% observed agreement with 50% chance agreement -> kappa 0
    assert cohens_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == pytest.approx(0.0)
    # textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)
