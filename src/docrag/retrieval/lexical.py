"""BM25 over chunk text with a code-aware tokenizer.

Docs questions often name identifiers (`PoolManager.initialize`, `sqrtPriceX96`,
`beforeSwap`). The tokenizer keeps each identifier whole *and* adds its camelCase /
dotted parts, so both "beforeSwap" and "before swap" match.
"""

from __future__ import annotations

import re
from pathlib import Path

import bm25s
import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = frozenset(
    (
        "a an and are as at be by for from has have how i if in into is it its of on or that "
        "the their then there these this to was what when where which who why will with you "
        "your do does can"
    ).split()
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        lw = word.lower()
        if lw not in _STOP:
            out.append(lw)
        parts = [p for seg in word.split(".") for p in _CAMEL_RE.findall(seg)]
        if len(parts) > 1 or "." in word:
            out.extend(p.lower() for p in parts if p.lower() not in _STOP)
    return out


class BM25Index:
    def __init__(self, ids: list[str], retriever: bm25s.BM25) -> None:
        self.ids = ids
        self._bm25 = retriever

    @classmethod
    def build(cls, ids: list[str], texts: list[str]) -> BM25Index:
        retriever = bm25s.BM25()
        retriever.index([tokenize(t) for t in texts], show_progress=False)
        return cls(ids, retriever)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        self._bm25.save(str(path))
        (path / "ids.txt").write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        ids = (path / "ids.txt").read_text(encoding="utf-8").split("\n")
        return cls(ids, bm25s.BM25.load(str(path)))
