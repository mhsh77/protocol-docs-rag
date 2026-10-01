"""Local sentence-transformers embedder with a persistent content-addressed cache.

Cache key = sha256(model name + text), so re-runs and different chunking configs that
share chunk texts never re-embed the same string.
"""

from __future__ import annotations

import hashlib
import sqlite3
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

from docrag.config import EmbeddingConfig

Vectors = npt.NDArray[np.float32]


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, Vectors]:
        out: dict[str, Vectors] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            for k, blob in self._db.execute(q, batch):
                out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, Vectors]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self._db.commit()


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, cache_path: Path | None) -> None:
        self.cfg = cfg
        self.cache = EmbeddingCache(cache_path) if cache_path else None

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]  # heavy import, loaded lazily
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.cfg.model, device="cpu")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.cfg.model}\x00{text}".encode()).hexdigest()

    def _encode(self, texts: list[str]) -> Vectors:
        vecs = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> Vectors:
        keys = [self._key(t) for t in texts]
        cached = self.cache.get_many(keys) if self.cache else {}
        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            new = self._encode([texts[i] for i in missing])
            fresh = {keys[i]: v for i, v in zip(missing, new, strict=True)}
            if self.cache:
                self.cache.put_many(fresh)
            cached.update(fresh)
        return np.stack([cached[k] for k in keys])

    def embed_query(self, text: str) -> Vectors:
        vec: Vectors = self._encode([self.cfg.query_instruction + text])[0]
        return vec
