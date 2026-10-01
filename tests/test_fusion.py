import pytest

from docrag.retrieval.fusion import reciprocal_rank_fusion as rrf


def ids(result: list[tuple[str, float]]) -> list[str]:
    return [d for d, _ in result]


def test_doc_ranked_well_in_both_lists_wins() -> None:
    dense = ["a", "b", "c"]
    bm25 = ["b", "d", "a"]
    out = rrf([dense, bm25], k=60)
    assert ids(out)[0] == "b"  # ranks 2 and 1 beat a's ranks 1 and 3
    assert set(ids(out)) == {"a", "b", "c", "d"}


def test_scores_match_formula() -> None:
    out = dict(rrf([["a", "b"], ["b"]], k=10))
    assert out["a"] == pytest.approx(1 / 11)
    assert out["b"] == pytest.approx(1 / 12 + 1 / 11)


def test_disjoint_lists_interleave_by_rank() -> None:
    out = rrf([["a1", "a2"], ["b1", "b2"]], k=60)
    # equal scores per rank; ties broken by best rank then id
    assert ids(out) == ["a1", "b1", "a2", "b2"]


def test_k_controls_top_rank_advantage() -> None:
    # "x" and "q" are each #1 in one list only; "y" is #3 in both lists.
    lists = [["x", "p", "y"], ["q", "r", "y"]]
    assert ids(rrf(lists, k=0))[:2] == ["q", "x"]  # small k: a single #1 dominates
    assert ids(rrf(lists, k=60))[0] == "y"  # large k: consistent agreement wins


def test_weights_and_duplicates() -> None:
    out = dict(rrf([["a", "a", "b"], ["b"]], k=0, weights=[1.0, 0.5]))
    assert out["a"] == pytest.approx(1.0)  # duplicate counted once at best rank
    assert out["b"] == pytest.approx(1 / 2 + 0.5)


def test_empty_and_invalid() -> None:
    assert rrf([[], []]) == []
    with pytest.raises(ValueError):
        rrf([["a"]], weights=[1.0, 2.0])
