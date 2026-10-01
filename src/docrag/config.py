"""Configuration: pipeline settings from YAML, secrets and model choices from the environment."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class PipelineConfig(BaseModel):
    corpus: CorpusConfig

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / self.corpus.name


class Settings(BaseSettings):
    """Runtime settings read from the environment / `.env`. Never hard-code secrets."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    corpus_config: Path = PROJECT_ROOT / "config" / "uniswap.yaml"


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    path = path or Settings().corpus_config
    with open(path, encoding="utf-8") as f:
        return PipelineConfig.model_validate(yaml.safe_load(f))
