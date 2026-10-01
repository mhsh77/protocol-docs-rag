"""Configuration: pipeline settings from YAML, secrets and model choices from the environment."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docrag.ingest.chunking import ChunkingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CorpusConfig(BaseModel):
    name: str
    repo: str
    commit: str
    license: str
    include_prefixes: list[str]
    exclude_prefixes: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=lambda: [".md", ".mdx"])
    strip_prefix: str = ""
    published_url_template: str
    license_file: str | None = "LICENSE"

    def wants(self, repo_path: str) -> bool:
        """Return True if a repo-relative path belongs in the corpus."""
        return (
            any(repo_path.startswith(p) for p in self.include_prefixes)
            and not any(repo_path.startswith(p) for p in self.exclude_prefixes)
            and any(repo_path.endswith(ext) for ext in self.extensions)
        )

    def published_url(self, repo_path: str) -> str:
        path = repo_path.removeprefix(self.strip_prefix)
        for ext in self.extensions:
            path = path.removesuffix(ext)
        path = path.removesuffix("/index").removesuffix("index")
        return self.published_url_template.format(path=path.rstrip("/"))

    def source_url(self, repo_path: str) -> str:
        return f"https://github.com/{self.repo}/blob/{self.commit}/{repo_path}"

    def raw_url(self, repo_path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{repo_path}"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    top_k: int = 5
    dense_candidates: int = 30
    bm25_candidates: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class ViolationPolicy(StrEnum):
    FLAG = "flag"  # return the answer, marked citation-invalid
    ABSTAIN = "abstain"  # replace the answer with an abstention


class GenerationConfig(BaseModel):
    protocol_name: str = "Uniswap"
    prompt_version: str = "answer_v2"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    citation_check: bool = True
    max_citation_retries: int = 1
    # Enforce a citation on every sentence (retry, then abstain). Off by default: in smoke
    # tests it turned correct false-premise corrections ("The premise is incorrect. ...")
    # into abstentions. Uncited sentences are still recorded and reported.
    require_sentence_coverage: bool = False
    on_violation: ViolationPolicy = ViolationPolicy.ABSTAIN
    # Abstain before calling the LLM when the top reranker score is below this.
    # None = disabled. Tuned on the dev split only (see docs/abstention-threshold.md).
    min_rerank_score: float | None = None


class RateLimits(BaseModel):
    per_user_per_minute: int = 4
    per_user_per_day: int = 30
    global_per_day: int = 150  # protects the free-tier LLM quota shared by all users


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    # BGE retrieval models expect this instruction on queries (not on passages).
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32


class PipelineConfig(BaseModel):
    corpus: CorpusConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    bot_limits: RateLimits = Field(default_factory=RateLimits)

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / self.corpus.name

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / self.corpus.name

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"


class Settings(BaseSettings):
    """Runtime settings read from the environment / `.env`. Never hard-code secrets."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    corpus_config: Path = PROJECT_ROOT / "config" / "uniswap.yaml"

    # gemini | groq | openrouter | ollama
    llm_provider: str = "groq"
    llm_base_url: str = ""  # override the provider's default endpoint
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    generator_model: str = ""
    judge_model: str = ""
    # Client-side throttle, kept under the provider's per-minute limit.
    llm_requests_per_minute: int = 25
    # Rolling-minute token budget (Groq free tier: 8K TPM per model). 0 = unlimited.
    llm_tokens_per_minute: int = 7500

    telegram_bot_token: str = ""
    # Optional override for the Qdrant directory (default: data/index/<corpus>/qdrant).
    qdrant_path: Path | None = None
    # Salt for hashing user ids in usage logs (raw Telegram ids are never written).
    log_salt: str = "change-me"

    @property
    def cache_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "cache"


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    path = path or Settings().corpus_config
    with open(path, encoding="utf-8") as f:
        return PipelineConfig.model_validate(yaml.safe_load(f))
