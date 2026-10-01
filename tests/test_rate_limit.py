import time

from docrag.llm.openai_compat import TokenRateLimiter


def test_token_limiter_allows_under_budget_without_waiting() -> None:
    lim = TokenRateLimiter(tpm=1000)
    lim.record(400)
    start = time.monotonic()
    lim.wait(500)
    assert time.monotonic() - start < 0.05


def test_token_limiter_blocks_when_budget_exceeded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    clock = [1000.0]
    slept: list[float] = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def fake_sleep(s: float) -> None:
        slept.append(s)
        clock[0] += s

    monkeypatch.setattr(time, "sleep", fake_sleep)
    lim = TokenRateLimiter(tpm=1000)
    lim.record(900)
    clock[0] += 10
    lim.wait(500)  # must wait until the 900-token event leaves the 60 s window
    assert sum(slept) >= 50
    assert clock[0] - 1000.0 > 60
