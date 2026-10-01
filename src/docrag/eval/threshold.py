"""Tune the retrieval-score abstention gate on the DEV split only.

For each candidate threshold t, simulate: abstain if the top reranker score < t, otherwise
keep whatever the system did (answer, or model abstention). Pick the threshold that
maximises correct abstentions minus false abstentions, subject to a cap on false
abstentions. The test split is never looked at here.
"""

from __future__ import annotations

from pydantic import BaseModel

from docrag.eval.runner import Record


class ThresholdPoint(BaseModel):
    threshold: float | None
    correct_abstention: float
    false_abstention: float
    n_unanswerable: int
    n_answerable: int

    @property
    def score(self) -> float:
        return self.correct_abstention - self.false_abstention


def simulate(records: list[Record], threshold: float | None) -> ThresholdPoint:
    def abstains(r: Record) -> bool:
        gated = (
            threshold is not None
            and r.top_rerank_score is not None
            and r.top_rerank_score < threshold
        )
        return gated or r.abstained

    unans = [r for r in records if r.expected == "abstain"]
    ans = [r for r in records if r.expected != "abstain"]
    return ThresholdPoint(
        threshold=threshold,
        correct_abstention=sum(abstains(r) for r in unans) / len(unans) if unans else 0.0,
        false_abstention=sum(abstains(r) for r in ans) / len(ans) if ans else 0.0,
        n_unanswerable=len(unans),
        n_answerable=len(ans),
    )


def sweep(records: list[Record], max_false_abstention: float = 0.10) -> list[ThresholdPoint]:
    scores = sorted({r.top_rerank_score for r in records if r.top_rerank_score is not None})
    candidates: list[float | None] = [None, *scores, *(s + 1e-6 for s in scores)]
    points = [simulate(records, t) for t in candidates]
    return sorted(
        points,
        key=lambda p: (p.false_abstention <= max_false_abstention, p.score),
        reverse=True,
    )


def best_threshold(records: list[Record], max_false_abstention: float = 0.10) -> ThresholdPoint:
    if any(r.split != "dev" for r in records):
        raise ValueError("threshold tuning must use dev-split records only")
    return sweep(records, max_false_abstention)[0]
