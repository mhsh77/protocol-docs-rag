"""Service layer (rate limiting, reply mapping, logging) and Telegram rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from docrag.bot.render import TELEGRAM_LIMIT, to_telegram_html
from docrag.config import RateLimits
from docrag.generation.assistant import AbstainReason, Answer, Citation
from docrag.service import AnswerService, Link, RateLimited, Reply, SlidingWindowLimiter, to_reply


def test_sliding_window_limiter() -> None:
    lim = SlidingWindowLimiter(limit=2, window_s=60)
    assert lim.check("u", now=0) == 0
    assert lim.check("u", now=1) == 0
    assert lim.check("u", now=2) == pytest.approx(58)  # third within a minute is refused
    assert lim.check("other", now=2) == 0  # per key
    assert lim.check("u", now=61) == 0  # first event left the window


def cite(alias: str, n: int) -> Citation:
    return Citation(alias=alias, chunk_id=f"d#{n}", heading=f"Doc > S{n}", url=f"https://x/{n}")


def test_to_reply_renumbers_citations_in_order() -> None:
    ans = Answer(
        question="q",
        text="A [S3]. B [S1][S3].",
        abstained=False,
        citations=[cite("S3", 3), cite("S1", 1)],
    )
    r = to_reply(ans)
    assert r.text == "A [1]. B [2][1]."
    assert [s.url for s in r.sources] == ["https://x/3", "https://x/1"]


def test_html_escapes_and_lists_sources() -> None:
    reply = Reply(
        text="Use <PoolManager> & hooks [1].",
        sources=[Link(label="Hooks > Intro", url="https://d/hooks?a=1&b=2")],
        abstained=False,
    )
    html = to_telegram_html(reply)
    assert "&lt;PoolManager&gt; &amp; hooks [1]" in html
    assert '<a href="https://d/hooks?a=1&amp;b=2">Hooks &gt; Intro</a>' in html


def test_html_abstention_shows_closest_section() -> None:
    reply = Reply(
        text="I couldn't find this.",
        sources=[],
        abstained=True,
        closest=Link(label="Fees", url="https://d/fees"),
    )
    html = to_telegram_html(reply)
    assert "Closest section" in html and 'href="https://d/fees"' in html
    assert "won't guess" in html


def test_html_respects_telegram_length_limit() -> None:
    reply = Reply(text="x" * 5000, sources=[Link(label="S", url="https://d")], abstained=False)
    html = to_telegram_html(reply)
    assert len(html) <= TELEGRAM_LIMIT
    assert "https://d" in html


class FakeAssistant:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, mode: Any = None) -> Answer:
        self.calls += 1
        return Answer(
            question=question,
            text="Not covered.",
            abstained=True,
            abstain_reason=AbstainReason.MODEL,
        )


def test_service_rate_limits_and_logs_without_raw_user_id(tmp_path: Path) -> None:
    log_path = tmp_path / "usage.jsonl"
    svc = AnswerService(
        FakeAssistant(),  # type: ignore[arg-type]
        limits=RateLimits(per_user_per_minute=1, per_user_per_day=10, global_per_day=10),
        log_path=log_path,
        user_salt="s",
    )
    assert svc.ask("12345", "q?", "telegram").abstained
    with pytest.raises(RateLimited) as e:
        svc.ask("12345", "again?", "telegram")
    assert e.value.scope == "user_minute"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["answer", "rate_limited"]
    assert records[0]["question"] == "q?" and records[0]["abstained"] is True
    assert "12345" not in log_path.read_text()  # user ids are hashed
