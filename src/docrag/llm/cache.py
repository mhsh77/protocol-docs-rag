"""Persistent response cache around any `LLMClient`.

Key = sha256 of everything that determines the request (model, system, prompt,
temperature, schema, max tokens). Effects:
* re-running the eval costs nothing and returns byte-identical outputs (provider-side
  nondeterminism cannot change a finished run);
* the "citation check off" configuration reuses the first-attempt answers of the
  "check on" run, because its single call is exactly the same request.
Cached responses keep the latency measured when they were first generated.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from docrag.llm.base import LLMClient, LLMResponse


class CachedLLM:
    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner = inner
        self.model = inner.model
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, resp TEXT)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        key = hashlib.sha256(
            json.dumps(
                [self.model, system, prompt, temperature, json_schema, max_output_tokens],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT resp FROM llm WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            return LLMResponse.model_validate_json(row[0])
        resp = self.inner.generate(
            prompt,
            system=system,
            temperature=temperature,
            json_schema=json_schema,
            max_output_tokens=max_output_tokens,
        )
        self.misses += 1
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, resp.model_dump_json())
            )
            self._db.commit()
        return resp
