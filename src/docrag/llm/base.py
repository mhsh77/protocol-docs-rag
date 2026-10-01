"""Provider-agnostic LLM interface. The rest of the code depends only on this."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    retries: int = 0


class LLMClient(Protocol):
    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Return the model's reply. With `json_schema`, the reply text is JSON."""
        ...
