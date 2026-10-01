"""Evaluation metrics. Pure functions over per-question records, so they are unit-tested
and can be recomputed from saved raw outputs in `eval/runs/` at any time."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np


def first_hit_rank(retrieved_ids: Sequence[str], gold_ids: set[str]) -> int | None:
    """1-based rank of the first retrieved id that is gold, or None."""
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_ids:
            return rank
    return None


def hit_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    rank = first_hit_rank(retrieved_ids, gold_ids)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: set[str], cutoff: int = 10) -> float:
    rank = first_hit_rank(retrieved_ids, gold_ids)
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def evidence_recall_at_k(
    retrieved_ids: Sequence[str], evidence_groups: Sequence[set[str]], k: int
) -> float:
    """Share of evidence pieces (each = set of chunks containing it) found in the top k.

    Multi-section questions have several evidence pieces; hit@k only asks for one of them,
    this asks for all of them.
    """
    if not evidence_groups:
        return 0.0
    top = set(retrieved_ids[:k])
    return sum(1.0 for g in evidence_groups if g & top) / len(evidence_groups)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for `stat` over per-question values."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of (b - a) over the same questions, with a paired bootstrap CI.

    Paired resampling (same question indices for both systems) is the right test when two
    configurations answer the same question set.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, mean, n_boot, alpha, seed)
    return (mean(diffs), lo, hi)


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items (chance-corrected agreement)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(labels_a)
    cats = set(labels_a) | set(labels_b)
    p_o = sum(x == y for x, y in zip(labels_a, labels_b, strict=True)) / n
    p_e = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in cats)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
