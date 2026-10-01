from pathlib import Path
from typing import Any

from docrag.llm.base import LLMResponse
from docrag.llm.cache import CachedLLM


class CountingLLM:
    model = "m"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **_: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=f"reply to {prompt}", model="m", latency_s=1.5)


def test_identical_requests_hit_cache_and_keep_original_latency(tmp_path: Path) -> None:
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path / "c.sqlite")
    a = llm.generate("q", system="s")
    b = llm.generate("q", system="s")
    assert inner.calls == 1 and llm.hits == 1
    assert a == b and b.latency_s == 1.5


def test_any_parameter_change_is_a_different_key(tmp_path: Path) -> None:
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path / "c.sqlite")
    llm.generate("q", system="s")
    llm.generate("q", system="s2")
    llm.generate("q", system="s", temperature=0.5)
    llm.generate("q", system="s", json_schema={"type": "object"})
    assert inner.calls == 4


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    inner = CountingLLM()
    CachedLLM(inner, tmp_path / "c.sqlite").generate("q")
    CachedLLM(inner, tmp_path / "c.sqlite").generate("q")
    assert inner.calls == 1


def test_provider_or_reasoning_setting_is_part_of_the_key(tmp_path: Path) -> None:
    a, b = CountingLLM(), CountingLLM()
    a.cache_tag = "https://provider-a|reasoning=default"  # type: ignore[attr-defined]
    b.cache_tag = "https://provider-a|reasoning=none"  # type: ignore[attr-defined]
    CachedLLM(a, tmp_path / "c.sqlite").generate("q")
    CachedLLM(b, tmp_path / "c.sqlite").generate("q")
    assert a.calls == 1 and b.calls == 1  # same model and prompt, different system: no hit
