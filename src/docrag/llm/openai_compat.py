"""`LLMClient` for any OpenAI-compatible chat endpoint (Groq, OpenRouter, Cerebras,
Ollama, vLLM, ...). Only `base_url`, key and model name differ between providers.

Structured output: we request `json_object` mode and put the JSON schema in the system
message, then validate the reply. This works on every compatible provider, whereas
strict `json_schema` mode is supported only by some models.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any

import openai
from openai import Omit, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject

from docrag.llm.base import LLMResponse
from docrag.llm.gemini import RateLimiter


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        rpm: int = 30,
        tpm: int | None = None,
        max_retries: int = 6,
        timeout_s: float = 120.0,
        max_wait_s: float = 300.0,
    ) -> None:
        self.model = model
        # SDK retries are disabled so that all retry/backoff behaviour lives here.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self._limiter = RateLimiter(rpm)
        self._tokens = TokenRateLimiter(tpm) if tpm else None
        self._max_retries = max_retries
        self._max_wait_s = max_wait_s
        self._max_json_retries = 2

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        sys_msg = system or ""
        if json_schema:
            sys_msg += (
                "\n\nRespond with a single JSON object that conforms to this JSON schema, "
                f"and nothing else:\n{json.dumps(json_schema)}"
            )
        messages: list[ChatCompletionMessageParam] = []
        if sys_msg.strip():
            messages.append({"role": "system", "content": sys_msg.strip()})
        messages.append({"role": "user", "content": prompt})

        response_format: ResponseFormatJSONObject | Omit = (
            {"type": "json_object"} if json_schema else omit
        )
        estimate = _estimate_tokens(sys_msg + prompt) + min(max_output_tokens, 400)
        attempt = 0
        json_failures = 0
        # Latency = provider time including provider-side retries/backoff (users feel those),
        # excluding our own client-side throttling (an artefact of free-tier quotas).
        t0 = time.perf_counter()
        throttle_s = 0.0
        while True:
            w0 = time.perf_counter()
            self._limiter.wait()
            if self._tokens:
                self._tokens.wait(estimate)
            throttle_s += time.perf_counter() - w0
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    response_format=response_format,
                )
            except openai.BadRequestError as e:
                # Groq validates JSON mode server-side and rejects invalid/truncated JSON with
                # 400 json_validate_failed. Retry, then return an empty reply that callers
                # treat as malformed output (counted, never silently dropped).
                if "json_validate_failed" not in str(e):
                    raise
                if self._tokens:
                    self._tokens.record(estimate)
                json_failures += 1
                if json_failures <= self._max_json_retries:
                    continue
                return LLMResponse(
                    text="",
                    model=self.model,
                    latency_s=time.perf_counter() - t0 - throttle_s,
                    throttle_s=throttle_s,
                    retries=attempt + json_failures,
                )
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                # Groq occasionally returns a bare 403 "Forbidden" that succeeds on retry
                # (observed during network/VPN changes). A real auth failure keeps failing
                # and is re-raised after max_retries.
                openai.PermissionDeniedError,
            ) as e:
                wait = _retry_after(e)
                if wait is not None and wait > self._max_wait_s:
                    # A long Retry-After means a daily quota, not a per-minute burst.
                    raise QuotaExhaustedError(self.model, wait) from e
                if attempt >= self._max_retries:
                    raise
                attempt += 1
                time.sleep(wait or min(60.0, 2.0**attempt))
                continue
            usage = resp.usage
            if self._tokens and usage:
                self._tokens.record(usage.total_tokens)
            return LLMResponse(
                text=resp.choices[0].message.content or "",
                model=self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_s=time.perf_counter() - t0 - throttle_s,
                throttle_s=throttle_s,
                retries=attempt + json_failures,
            )


class QuotaExhaustedError(RuntimeError):
    """The provider's daily quota is used up; callers should save progress and stop."""

    def __init__(self, model: str, retry_after_s: float) -> None:
        super().__init__(
            f"Daily quota exhausted for {model}; retry in ~{retry_after_s / 60:.0f} min"
        )
        self.model = model
        self.retry_after_s = retry_after_s


class TokenRateLimiter:
    """Keep total tokens in any rolling 60 s window under `tpm` (thread-safe)."""

    def __init__(self, tpm: int) -> None:
        self.tpm = tpm
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _used(self, now: float) -> int:
        while self._events and now - self._events[0][0] > 60.0:
            self._events.popleft()
        return sum(n for _, n in self._events)

    def wait(self, estimate: int) -> None:
        estimate = min(estimate, self.tpm)
        while True:
            with self._lock:
                now = time.monotonic()
                if self._used(now) + estimate <= self.tpm or not self._events:
                    return
                sleep_for = 60.0 - (now - self._events[0][0]) + 0.1
            time.sleep(max(0.1, sleep_for))

    def record(self, tokens: int) -> None:
        with self._lock:
            self._events.append((time.monotonic(), tokens))


def _estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1  # conservative: ~3 chars/token for English + code


def _retry_after(e: Exception) -> float | None:
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None
