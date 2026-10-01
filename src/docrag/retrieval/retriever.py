"""Switchable retrieval: `dense` (naive baseline), `hybrid` (dense + BM25 via RRF),
`hybrid_rerank` (hybrid candidates re-scored by a cross-encoder)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from docrag.config import PipelineConfig, RetrievalConfig, RetrievalMode, Settings
from docrag.ingest.chunking import Chunk
from docrag.ingest.pipeline import read_chunks
from docrag.retrieval.embedder import Embedder
from docrag.retrieval.fusion import reciprocal_rank_fusion
from docrag.retrieval.lexical import BM25Index
from docrag.retrieval.reranker import CrossEncoderReranker
from docrag.retrieval.vector_store import VectorStore


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine (dense), RRF score (hybrid) or cross-encoder logit (rerank)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


def index_dir(cfg: PipelineConfig) -> Path:
    from docrag.config import PROJECT_ROOT

    return PROJECT_ROOT / "data" / "index" / cfg.corpus.name


def build_index(cfg: PipelineConfig, settings: Settings | None = None) -> int:
    """Embed chunks (cached), (re)build the Qdrant collection and the BM25 index."""
    settings = settings or Settings()
    chunks = read_chunks(cfg.chunks_path)
    ids = [c.chunk_id for c in chunks]
    texts = [c.embed_text() for c in chunks]
    embedder = Embedder(cfg.embedding, settings.cache_dir / "embeddings.sqlite")
    vectors = embedder.embed_passages(texts)
    out = index_dir(cfg)
    store = VectorStore(out / "qdrant")
    store.rebuild(ids, vectors)
    store.close()
    (out / "bm25").mkdir(parents=True, exist_ok=True)
    BM25Index.build(ids, texts).save(out / "bm25")
    meta = {
        "n_chunks": len(chunks),
        "embedding_model": cfg.embedding.model,
        "chunking": cfg.chunking.model_dump(),
        "corpus_commit": cfg.corpus.commit,
    }
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return len(chunks)


class Retriever:
    def __init__(
        self,
        cfg: PipelineConfig,
        rcfg: RetrievalConfig,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or Settings()
        self.rcfg = rcfg
        self.chunks = {c.chunk_id: c for c in read_chunks(cfg.chunks_path)}
        idx = index_dir(cfg)
        meta = json.loads((idx / "index_meta.json").read_text(encoding="utf-8"))
        if meta["n_chunks"] != len(self.chunks) or meta["chunking"] != cfg.chunking.model_dump():
            raise RuntimeError("Index is stale relative to chunks/config; run `rag ingest`.")
        self.embedder = Embedder(cfg.embedding, settings.cache_dir / "embeddings.sqlite")
        # QDRANT_PATH lets a second process read a copy (embedded Qdrant holds a file lock).
        self.store = VectorStore(settings.qdrant_path or idx / "qdrant")
        self.bm25 = BM25Index.load(idx / "bm25")
        self._reranker: CrossEncoderReranker | None = None

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.rcfg.reranker_model)
        return self._reranker

    def retrieve(self, query: str, mode: RetrievalMode | None = None) -> list[RetrievedChunk]:
        mode = mode or self.rcfg.mode
        r = self.rcfg
        qvec = self.embedder.embed_query(query)
        n_dense = r.top_k if mode is RetrievalMode.DENSE else r.dense_candidates
        dense = self.store.search(qvec, n_dense)
        dense_rank = {cid: i for i, (cid, _) in enumerate(dense, 1)}
        if mode is RetrievalMode.DENSE:
            return [
                RetrievedChunk(chunk=self.chunks[cid], score=s, dense_rank=dense_rank[cid])
                for cid, s in dense
            ]

        lexical = self.bm25.search(query, r.bm25_candidates)
        bm25_rank = {cid: i for i, (cid, _) in enumerate(lexical, 1)}
        fused = reciprocal_rank_fusion([[c for c, _ in dense], [c for c, _ in lexical]], k=r.rrf_k)
        n_keep = r.top_k if mode is RetrievalMode.HYBRID else r.rerank_candidates
        candidates = [
            RetrievedChunk(
                chunk=self.chunks[cid],
                score=s,
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
            )
            for cid, s in fused[:n_keep]
        ]
        if mode is RetrievalMode.HYBRID:
            return candidates

        scores = self.reranker.score(query, [c.chunk.embed_text() for c in candidates])
        for c, s in zip(candidates, scores, strict=True):
            c.rerank_score = s
            c.score = s
        candidates.sort(key=lambda c: -c.score)
        return candidates[: r.top_k]

    def close(self) -> None:
        self.store.close()
