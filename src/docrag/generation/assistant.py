"""The transport-agnostic assistant: question in, grounded answer with citations out.

Telegram (and any future Discord/web adapter) only calls `Assistant.answer`.

Abstention is enforced in two layers:
1. Retrieval gate (optional): if the best reranker score is below a threshold tuned on
   the dev split, we abstain without calling the LLM.
2. Model decision: the prompt requires an explicit `abstain` field; the model must abstain
   when the excerpts do not support an answer.
A third safety net: if citations are still invalid after one retry, the configured
policy either flags the answer or converts it into an abstention.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum

from pydantic import BaseModel, Field

from docrag.config import GenerationConfig, RetrievalMode, ViolationPolicy
from docrag.generation.citations import CitationCheck, check_citations
from docrag.generation.prompt import ANSWER_SCHEMA, alias, build_prompt, load_template
from docrag.llm.base import LLMClient, LLMResponse
from docrag.retrieval.retriever import RetrievedChunk, Retriever


class Citation(BaseModel):
    alias: str
    chunk_id: str
    heading: str
    url: str


class AbstainReason(StrEnum):
    MODEL = "model"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    CITATION_VIOLATION = "citation_violation"
    MALFORMED_OUTPUT = "malformed_output"


class Answer(BaseModel):
    question: str
    text: str
    abstained: bool
    abstain_reason: AbstainReason | None = None
    citations: list[Citation] = Field(default_factory=list)
    closest: Citation | None = None  # pointer offered when abstaining
    citation_check: CitationCheck | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    llm_calls: list[LLMResponse] = Field(default_factory=list)
    retrieval_s: float = 0.0
    total_s: float = 0.0
    raw_model_text: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)


ABSTAIN_TEXT = "I couldn't find this in the {protocol} documentation, so I won't guess."


class Assistant:
    def __init__(self, retriever: Retriever, llm: LLMClient, cfg: GenerationConfig) -> None:
        self.retriever = retriever
        self.llm = llm
        self.cfg = cfg
        self.template = load_template(cfg.prompt_version)

    def answer(self, question: str, mode: RetrievalMode | None = None) -> Answer:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, mode)
        retrieval_s = time.perf_counter() - t0
        return self.answer_from(question, retrieved, retrieval_s=retrieval_s, t0=t0)

    def answer_from(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        retrieval_s: float = 0.0,
        t0: float | None = None,
    ) -> Answer:
        """Generate from an already-retrieved context (lets the eval reuse retrieval)."""
        t0 = t0 if t0 is not None else time.perf_counter()
        cfg = self.cfg
        base = Answer(question=question, text="", abstained=False, retrieved=retrieved)
        base.retrieval_s = retrieval_s

        top = retrieved[0] if retrieved else None
        if (
            cfg.min_rerank_score is not None
            and top is not None
            and top.rerank_score is not None
            and top.rerank_score < cfg.min_rerank_score
        ):
            return self._finish(
                self._abstain(base, AbstainReason.LOW_RETRIEVAL_SCORE, closest_idx=0), t0
            )
        if not retrieved:
            return self._finish(self._abstain(base, AbstainReason.LOW_RETRIEVAL_SCORE), t0)

        system, user = build_prompt(question, retrieved, cfg.protocol_name, self.template)
        attempts = 1 + (cfg.max_citation_retries if cfg.citation_check else 0)
        parsed: dict[str, object] | None = None
        check: CitationCheck | None = None
        for attempt in range(attempts):
            resp = self.llm.generate(
                user,
                system=system,
                temperature=cfg.temperature,
                json_schema=ANSWER_SCHEMA,
                max_output_tokens=cfg.max_output_tokens,
            )
            base.llm_calls.append(resp)
            base.raw_model_text = resp.text
            parsed = _parse(resp.text)
            if parsed is None:
                continue
            abstained = bool(parsed.get("abstain"))
            check = check_citations(
                str(parsed.get("answer", "")),
                len(retrieved),
                abstained,
                require_sentence_coverage=cfg.require_sentence_coverage,
            )
            if not cfg.citation_check or check.valid:
                break
            if attempt + 1 < attempts:
                user = _with_feedback(user, check, len(retrieved))

        if parsed is None:
            return self._finish(self._abstain(base, AbstainReason.MALFORMED_OUTPUT), t0)

        text = str(parsed.get("answer", "")).strip()
        base.citation_check = check
        if parsed.get("abstain"):
            idx = _alias_index(parsed.get("closest_source"), len(retrieved))
            return self._finish(
                self._abstain(base, AbstainReason.MODEL, closest_idx=idx, text=text or None), t0
            )
        if (
            cfg.citation_check
            and check is not None
            and not check.valid
            and cfg.on_violation is ViolationPolicy.ABSTAIN
        ):
            return self._finish(
                self._abstain(base, AbstainReason.CITATION_VIOLATION, closest_idx=0), t0
            )

        base.text = text
        base.citations = [
            _citation(retrieved, i)
            for a in (check.cited if check else [])
            if (i := _alias_index(a, len(retrieved))) is not None
        ]
        return self._finish(base, t0)

    def _abstain(
        self,
        ans: Answer,
        reason: AbstainReason,
        closest_idx: int | None = None,
        text: str | None = None,
    ) -> Answer:
        ans.abstained = True
        ans.abstain_reason = reason
        ans.text = text or ABSTAIN_TEXT.format(protocol=self.cfg.protocol_name)
        if closest_idx is not None and closest_idx < len(ans.retrieved):
            ans.closest = _citation(ans.retrieved, closest_idx)
        ans.citations = []
        return ans

    @staticmethod
    def _finish(ans: Answer, t0: float) -> Answer:
        ans.total_s = time.perf_counter() - t0
        return ans


def _parse(text: str) -> dict[str, object] | None:
    text = text.strip()
    if text.startswith("```"):  # tolerate fenced JSON
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "answer" not in data or "abstain" not in data:
        return None
    return data


def _alias_index(value: object, n: int) -> int | None:
    if not isinstance(value, str) or not value.strip().upper().startswith("S"):
        return None
    try:
        i = int(value.strip().upper().removeprefix("S")) - 1
    except ValueError:
        return None
    return i if 0 <= i < n else None


def _citation(retrieved: list[RetrievedChunk], i: int) -> Citation:
    c = retrieved[i].chunk
    return Citation(alias=alias(i), chunk_id=c.chunk_id, heading=c.heading_str, url=c.url)


def _with_feedback(user: str, check: CitationCheck, n: int) -> str:
    return (
        f"{user}\n\nYour previous answer failed the citation check: {check.reason}. "
        f"Only S1..S{n} exist. Every factual sentence must end with a citation to a source "
        "that supports it. If you cannot support the answer, set abstain to true."
    )
