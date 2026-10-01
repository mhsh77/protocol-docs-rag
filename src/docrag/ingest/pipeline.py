"""Raw docs -> normalized Markdown -> chunks.jsonl."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from docrag.config import PipelineConfig
from docrag.corpus.fetch import MANIFEST_NAME, read_manifest
from docrag.ingest.chunking import Chunk, ChunkingConfig, TokenCounter, chunk_document
from docrag.ingest.mdx import normalize_mdx


@lru_cache(maxsize=4)
def hf_token_counter(model_name: str) -> TokenCounter:
    """Count tokens with the embedding model's own tokenizer, so budgets match what it sees."""
    tok = Tokenizer.from_pretrained(model_name)
    tok.no_truncation()

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return count


def build_chunks(cfg: PipelineConfig, chunking: ChunkingConfig | None = None) -> list[Chunk]:
    chunking = chunking or cfg.chunking
    count = hf_token_counter(cfg.embedding.model)
    chunks: list[Chunk] = []
    for entry in read_manifest(cfg.raw_dir / MANIFEST_NAME):
        source = (cfg.raw_dir / entry.local_path).read_text(encoding="utf-8")
        doc = normalize_mdx(source)
        slug = entry.local_path.rsplit(".", 1)[0]
        chunks.extend(
            chunk_document(
                text=doc.text,
                title=doc.title,
                doc_id=entry.doc_id,
                doc_slug=slug,
                url=entry.published_url,
                source_url=entry.source_url,
                cfg=chunking,
                count=count,
            )
        )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.model_validate(json.loads(line)) for line in f if line.strip()]
