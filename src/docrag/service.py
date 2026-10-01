"""Transport-agnostic answering service used by every chat adapter (Telegram today,
Discord or web later). Adapters only translate messages in and `Reply` objects out.

Responsibilities kept here, not in adapters:
* building the assistant from config,
* per-user and global rate limits,
* structured usage logging (question, retrieved chunk ids, answer, latency),
* turning an `Answer` into a transport-neutral `Reply`.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import structlog
from pydantic import BaseModel

from docrag.config import PROJECT_ROOT, PipelineConfig, RateLimits, Settings
from docrag.generation.assistant import Answer, Assistant
from docrag.llm import make_client
from docrag.llm.cache import CachedLLM
from docrag.retrieval.retriever import Retriever


class Link(BaseModel):
    label: str
    url: str


class Reply(BaseModel):
    text: str  # plain text, inline markers like [1] refer to `sources`
    sources: list[Link]
    abstained: bool
    closest: Link | None = None


class RateLimited(Exception):
    def __init__(self, retry_after_s: float, scope: str) -> None:
        super().__init__(f"rate limited ({scope}), retry in {retry_after_s:.0f}s")
        self.retry_after_s = retry_after_s
        self.scope = scope


class SlidingWindowLimiter:
    """At most `limit` events per `window_s` per key."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float:
        """Record an event and return 0, or return seconds to wait (event not recorded)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._events[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return self.window_s - (now - q[0])
            q.append(now)
            return 0.0


def to_reply(ans: Answer) -> Reply:
    """Renumber cited sources [S2][S5] -> [1][2] in citation order for display."""
    text = ans.text
    sources: list[Link] = []
    for i, c in enumerate(ans.citations, 1):
        text = text.replace(f"[{c.alias}]", f"[{i}]")
        sources.append(Link(label=c.heading, url=c.url))
    closest = Link(label=ans.closest.heading, url=ans.closest.url) if ans.closest else None
    return Reply(text=text, sources=sources, abstained=ans.abstained, closest=closest)


def _hash_user(user_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:16]


class AnswerService:
    def __init__(
        self,
        assistant: Assistant,
        limits: RateLimits | None = None,
        log_path: Path | None = None,
        user_salt: str = "",
    ) -> None:
        self.assistant = assistant
        self.limits = limits or RateLimits()
        self._minute = SlidingWindowLimiter(self.limits.per_user_per_minute, 60)
        self._day = SlidingWindowLimiter(self.limits.per_user_per_day, 86_400)
        self._global = SlidingWindowLimiter(self.limits.global_per_day, 86_400)
        self._salt = user_salt
        self._log = _usage_logger(log_path or PROJECT_ROOT / "logs" / "usage.jsonl")

    def ask(self, user_id: str, question: str, channel: str) -> Reply:
        uid = _hash_user(user_id, self._salt)
        for limiter, key, scope in (
            (self._minute, uid, "user_minute"),
            (self._day, uid, "user_day"),
            (self._global, "*", "global_day"),
        ):
            wait = limiter.check(key)
            if wait > 0:
                self._log.info("rate_limited", user=uid, channel=channel, scope=scope)
                raise RateLimited(wait, scope)
        ans = self.assistant.answer(question)
        self._log.info(
            "answer",
            user=uid,
            channel=channel,
            question=question,
            retrieved=[r.chunk.chunk_id for r in ans.retrieved],
            cited=[c.chunk_id for c in ans.citations],
            answer=ans.text,
            abstained=ans.abstained,
            abstain_reason=ans.abstain_reason,
            citation_valid=ans.citation_check.valid if ans.citation_check else None,
            retrieval_s=round(ans.retrieval_s, 3),
            total_s=round(ans.total_s, 3),
            input_tokens=ans.input_tokens,
            output_tokens=ans.output_tokens,
        )
        return to_reply(ans)


def _usage_logger(path: Path) -> structlog.stdlib.BoundLogger:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Kept open for the process lifetime; line-buffered so each record is flushed.
    fh = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    return structlog.wrap_logger(  # type: ignore[no-any-return]
        structlog.PrintLogger(fh),
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


def build_service(cfg: PipelineConfig, settings: Settings) -> AnswerService:
    retriever = Retriever(cfg, cfg.retrieval, settings)
    llm = CachedLLM(
        make_client(settings.generator_model, settings), settings.cache_dir / "llm_cache.sqlite"
    )
    return AnswerService(
        Assistant(retriever, llm, cfg.generation),
        limits=cfg.bot_limits,
        user_salt=settings.log_salt,
    )
