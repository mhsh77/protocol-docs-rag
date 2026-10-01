import pytest

from docrag.eval.build_set import apply_curation, assign_splits
from docrag.eval.dataset import Category, EvalQuestion, Expected


def q(qid: str, cat: Category = Category.FACTUAL) -> EvalQuestion:
    return EvalQuestion(
        id=qid, split="", category=cat, question=f"question {qid}?", expected=Expected.ANSWER
    )


def test_drop_rewrite_recategorize() -> None:
    qs = [q("d1"), q("d2"), q("d3", Category.NUMERIC)]
    out = apply_curation(
        qs,
        {
            "drop": {"d1": "bad"},
            "rewrite": {"d2": {"question": "Better question?", "reference": "Ref."}},
            "recategorize": {"d3": "factual"},
        },
    )
    assert [x.id for x in out] == ["d2", "d3"]
    assert out[0].question == "Better question?" and out[0].reference_answer == "Ref."
    assert out[0].source == "llm_draft+human_edit"
    assert out[1].category is Category.FACTUAL


def test_unknown_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        apply_curation([q("d1")], {"drop": {"nope": "x"}, "rewrite": {}, "recategorize": {}})


def test_splits_are_stratified_and_deterministic() -> None:
    qs = [q(f"f{i}") for i in range(10)] + [q(f"n{i}", Category.NUMERIC) for i in range(10)]
    a, b = assign_splits(qs), assign_splits(qs)
    assert [(x.question, x.split) for x in a] == [(x.question, x.split) for x in b]
    for cat in (Category.FACTUAL, Category.NUMERIC):
        assert sum(x.split == "dev" for x in a if x.category is cat) == 3  # round(10 * 0.27)
    assert [x.id for x in a] == [f"q{i:03d}" for i in range(1, 21)]
