"""LLM clients. `make_client` is the only place that knows about concrete providers."""

from __future__ import annotations

from docrag.config import Settings
from docrag.llm.base import LLMClient, LLMResponse

# Providers reachable through the OpenAI-compatible adapter: name -> base URL.
OPENAI_COMPAT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "ollama": "http://localhost:11434/v1",
}


def make_client(
    model: str,
    settings: Settings | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    tokens_per_minute: int | None = None,
) -> LLMClient:
    """Build a client. `provider` overrides LLM_PROVIDER (e.g. a separate judge provider)."""
    settings = settings or Settings()
    provider = provider or settings.llm_provider
    if provider == "gemini":
        from docrag.llm.gemini import GeminiClient

        return GeminiClient(
            model=model,
            api_key=_require(settings.gemini_api_key, "GEMINI_API_KEY"),
            rpm=settings.llm_requests_per_minute,
        )
    if provider in OPENAI_COMPAT_BASE_URLS:
        from docrag.llm.openai_compat import OpenAICompatClient

        key = {
            "groq": settings.groq_api_key,
            "openrouter": settings.openrouter_api_key,
            "cerebras": settings.cerebras_api_key,
        }.get(provider, "unused")
        return OpenAICompatClient(
            model=model,
            api_key=_require(key, f"{provider.upper()}_API_KEY"),
            base_url=settings.llm_base_url or OPENAI_COMPAT_BASE_URLS[provider],
            rpm=settings.llm_requests_per_minute,
            tpm=(tokens_per_minute or settings.llm_tokens_per_minute) or None,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _require(value: str, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set (see .env.example).")
    return value


__all__ = ["LLMClient", "LLMResponse", "make_client"]
