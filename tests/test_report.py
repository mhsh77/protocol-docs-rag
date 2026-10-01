"""Semantics of the headline metrics, on hand-built records."""

from __future__ import annotations

from typing import Any

import pytest

from docrag.eval.judge import Claim, Verdict
from docrag.eval.report import compare, per_question, summarize
from docrag.eval.runner import Record


def rec(qid: str = "q1", **kw: Any) -> Record:
    base: dict[str, Any] = dict(
        qid=qid,
        config="c",
        split="test",
        category="factual",
        subtype=None,
        expected="answer",
        question="?",
        reference="ref",
        retrieved_ids=["a", "b", "c", "d", "e"],
        gold_groups=[["c"]],
        answer="x [S1].",
        abstained=False,
        abstain_reason=None,
        cited_ids=["a"],
        citation_valid=True,
        citation_first_attempt_valid=True,
        n_llm_calls=1,
        input_tokens=100,
        output_tokens=20,
        retrieval_s=0.5,
        generation_s=1.0,
    )
    base.update(kw)
    return Record(**base)


def verdict(correctness: int, supported: list[bool]) -> Verdict:
    return Verdict(
        correctness=correctness,
        claims=[Claim(claim=f"c{i}", supported=s) for i, s in enumerate(supported)],
    )


def test_answered_question_metrics() -> None:
    m = per_question(rec(verdict=verdict(2, [True, False])))
    assert m["hit@1"] == 0.0 and m["hit@5"] == 1.0
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert m["correctness"] == 1.0 and m["fully_correct"] == 1.0
    assert m["unsupported_claims"] == 1.0 and m["total_claims"] == 2.0
    assert m["false_abstention"] == 0.0
    assert m["correct_abstention"] is None  # not an unanswerable question


def test_abstaining_on_answerable_counts_as_wrong_and_false_abstention() -> None:
    m = per_question(rec(abstained=True, answer="I don't know", verdict=None))
    assert m["correctness"] == 0.0
    assert m["false_abstention"] == 1.0
    assert m["unsupported_claims"] is None  # no claims made, nothing to hallucinate
    assert m["citation_valid"] is None


def test_unanswerable_question_metrics() -> None:
    declined = per_question(rec(expected="abstain", gold_groups=[], abstained=True))
    answered = per_question(rec(expected="abstain", gold_groups=[], verdict=verdict(0, [False])))
    assert declined["correct_abstention"] == 1.0
    assert answered["correct_abstention"] == 0.0
    assert declined["hit@5"] is None and declined["correctness"] is None
    assert answered["any_unsupported"] == 1.0


def test_hallucination_rate_is_micro_averaged_over_claims() -> None:
    s = summarize(
        [
            rec("q1", verdict=verdict(2, [True, True, True])),
            rec("q2", verdict=verdict(1, [False])),
            rec("q3", abstained=True, verdict=None),
        ]
    )
    hr = s["hallucination_rate"]
    assert isinstance(hr, dict)
    assert hr["unsupported_claims"] == 1 and hr["total_claims"] == 4
    assert hr["mean"] == pytest.approx(0.25)


def test_paired_compare_uses_only_common_applicable_questions() -> None:
    base = [rec("q1", retrieved_ids=["z"]), rec("q2", retrieved_ids=["z"])]
    better = [rec("q1"), rec("q2")]  # gold "c" at rank 3 for both
    diff = compare(base, better, "hit@5")
    assert diff is not None
    assert diff[0] == pytest.approx(1.0)


def test_update_readme_block_replaces_only_generated_section(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from docrag.eval.report import README_END, README_START, update_readme_block

    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\nintro\n{README_START}\nold table\n{README_END}\n\nrest\n")
    update_readme_block(readme, "| new | table |\n")
    text = readme.read_text()
    assert "old table" not in text and "| new | table |" in text
    assert text.startswith("# T\n\nintro\n") and text.endswith(f"{README_END}\n\nrest\n")
