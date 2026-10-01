"""Cross-encoder reranker: scores (query, passage) pairs jointly, unlike bi-encoders."""

from __future__ import annotations

from functools import cached_property


class CrossEncoderReranker:
    def __init__(self, model: str, max_length: int = 384, batch_size: int = 16) -> None:
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]  # heavy import, loaded lazily
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, max_length=self.max_length, device="cpu")

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            activation_fn=None,
        )
        return [float(s) for s in scores]
