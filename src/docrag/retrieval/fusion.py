"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum_i  w_i / (k + rank_i(d)),   rank starting at 1.

RRF uses only ranks, so dense cosine scores and BM25 scores (different scales) can be
fused without calibration. `k` damps the advantage of top ranks; 60 is the paper's value.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Returns (id, score) sorted by score desc.

    Ties are broken by best single-list rank, then by id, so output is deterministic.
    Duplicate ids within one ranking count once, at their best rank.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        unique = list(dict.fromkeys(ranking))  # dedupe, keeping first (best) position
        for rank, doc_id in enumerate(unique, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))
