"""Gemini implementation of `LLMClient` with client-side rate limiting and retries.

The free tier enforces per-minute and per-day request limits, so every call goes
through a shared token-bucket limiter and 429/503 responses are retried with backoff.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from google import genai
from google.genai import errors, types

from docrag.llm.base import LLMResponse

_RETRYABLE = {429, 500, 502, 503, 504}


class RateLimiter:
    """Allow at most `rpm` calls per rolling minute (thread-safe)."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class GeminiClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        rpm: int = 10,
        max_retries: int = 6,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        self._limiter = RateLimiter(rpm)
        self._max_retries = max_retries

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_schema else None,
            response_json_schema=json_schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        attempt = 0
        while True:
            self._limiter.wait()
            start = time.perf_counter()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except errors.APIError as e:
                # Google's front end occasionally returns a 403 *HTML* page under load;
                # a real permission error comes back as JSON with a PERMISSION_DENIED status.
                transient_403 = e.code == 403 and "<html" in str(e.message or "").lower()
                retryable = e.code in _RETRYABLE or transient_403
                if retryable and attempt < self._max_retries:
                    attempt += 1
                    time.sleep(min(60.0, 2.0**attempt))
                    continue
                raise
            usage = resp.usage_metadata
            return LLMResponse(
                text=resp.text or "",
                model=self.model,
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=(usage.candidates_token_count or 0) if usage else 0,
                latency_s=time.perf_counter() - start,
                retries=attempt,
            )
