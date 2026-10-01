"""Dense index in Qdrant *local mode* (embedded, file-backed, no server or account).

Same client API as a Qdrant server, so moving to a hosted/server Qdrant is a
constructor change. Note: local mode takes a file lock, so one process at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from docrag.retrieval.embedder import Vectors

COLLECTION = "chunks"


class VectorStore:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))

    def rebuild(self, ids: list[str], vectors: Vectors) -> None:
        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)
        self._client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=int(vectors.shape[1]), distance=models.Distance.COSINE
            ),
        )
        self._client.upload_points(
            COLLECTION,
            points=[
                models.PointStruct(id=i, vector=v.tolist(), payload={"chunk_id": cid})
                for i, (cid, v) in enumerate(zip(ids, vectors, strict=True))
            ],
            batch_size=256,
        )

    def search(self, query_vector: Vectors, k: int) -> list[tuple[str, float]]:
        hits = self._client.query_points(
            COLLECTION, query=np.asarray(query_vector).tolist(), limit=k, with_payload=True
        ).points
        return [(str(h.payload["chunk_id"]), float(h.score)) for h in hits if h.payload]

    def count(self) -> int:
        if not self._client.collection_exists(COLLECTION):
            return 0
        return self._client.count(COLLECTION).count

    def close(self) -> None:
        self._client.close()
