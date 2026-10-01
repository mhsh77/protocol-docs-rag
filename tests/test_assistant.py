"""Assistant control flow with a scripted fake LLM (no network)."""

from __future__ import annotations

import json
from typing import Any

from docrag.config import GenerationConfig, ViolationPolicy
from docrag.generation.assistant import AbstainReason, Assistant
from docrag.ingest.chunking import Chunk
from docrag.llm.base import LLMResponse
from docrag.retrieval.retriever import RetrievedChunk


class FakeLLM:
    model = "fake"

    def __init__(self, replies: list[dict[str, Any] | str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_: Any) -> LLMResponse:
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        return LLMResponse(text=r if isinstance(r, str) else json.dumps(r), model="fake")


def retrieved(n: int = 3, rerank: float | None = None) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"doc#{i:03d}",
                doc_id="content/doc.mdx",
                url=f"https://example.org/docs/doc{i}",
                source_url="https://github.com/x",
                heading_path=["Doc", f"Section {i}"],
                text=f"Fact number {i}.",
                n_tokens=4,
                ordinal=i,
            ),
            score=1.0,
            rerank_score=rerank,
        )
        for i in range(n)
    ]


def make(replies: list[dict[str, Any] | str], **cfg: Any) -> tuple[Assistant, FakeLLM]:
    llm = FakeLLM(replies)
    return Assistant(retriever=None, llm=llm, cfg=GenerationConfig(**cfg)), llm  # type: ignore[arg-type]


def ok(answer: str) -> dict[str, Any]:
    return {"abstain": False, "answer": answer, "closest_source": None}


def test_valid_answer_maps_aliases_to_chunks() -> None:
    a, _ = make([ok("Fact one holds [S1]. Fact three also holds [S3].")])
    ans = a.answer_from("q", retrieved())
    assert not ans.abstained
    assert [c.chunk_id for c in ans.citations] == ["doc#000", "doc#002"]
    assert ans.citation_check is not None and ans.citation_check.valid


def test_invalid_citation_retried_once_with_feedback_then_accepted() -> None:
    a, llm = make([ok("Fact holds here [S9]."), ok("Fact holds here [S2].")])
    ans = a.answer_from("q", retrieved())
    assert len(llm.prompts) == 2
    assert "failed the citation check" in llm.prompts[1]
    assert not ans.abstained and ans.citations[0].chunk_id == "doc#001"


def test_persistent_violation_becomes_abstention_by_default() -> None:
    a, _ = make([ok("Unsupported claim here [S9]."), ok("Still unsupported claim here.")])
    ans = a.answer_from("q", retrieved())
    assert ans.abstained and ans.abstain_reason is AbstainReason.CITATION_VIOLATION
    assert ans.closest is not None and ans.closest.chunk_id == "doc#000"


def test_flag_policy_keeps_answer_but_marks_invalid() -> None:
    a, _ = make(
        [ok("Unsupported claim here [S9]."), ok("Again unsupported [S7].")],
        on_violation=ViolationPolicy.FLAG,
    )
    ans = a.answer_from("q", retrieved())
    assert not ans.abstained
    assert ans.citation_check is not None and not ans.citation_check.valid


def test_citation_check_off_makes_single_call_and_keeps_answer() -> None:
    a, llm = make([ok("Unsupported claim here [S9].")], citation_check=False)
    ans = a.answer_from("q", retrieved())
    assert len(llm.prompts) == 1
    assert not ans.abstained
    assert ans.citation_check is not None and not ans.citation_check.valid  # still measured


def test_model_abstention_points_to_closest_source() -> None:
    reply = {"abstain": True, "answer": "Not covered.", "closest_source": "S2"}
    a, _ = make([reply])
    ans = a.answer_from("q", retrieved())
    assert ans.abstained and ans.abstain_reason is AbstainReason.MODEL
    assert ans.closest is not None and ans.closest.chunk_id == "doc#001"


def test_low_rerank_score_abstains_without_llm_call() -> None:
    a, llm = make([], min_rerank_score=0.0)
    ans = a.answer_from("q", retrieved(rerank=-3.2))
    assert ans.abstained and ans.abstain_reason is AbstainReason.LOW_RETRIEVAL_SCORE
    assert llm.prompts == []


def test_malformed_output_abstains() -> None:
    a, _ = make(["not json", "still not json"])
    ans = a.answer_from("q", retrieved())
    assert ans.abstained and ans.abstain_reason is AbstainReason.MALFORMED_OUTPUT


def test_prompt_contains_sources_and_question() -> None:
    a, llm = make([ok("Fact holds [S1].")])
    a.answer_from("What is fact one?", retrieved(2))
    p = llm.prompts[0]
    assert "Question: What is fact one?" in p
    assert "[S1] Doc > Section 0" in p and "URL: https://example.org/docs/doc1" in p
