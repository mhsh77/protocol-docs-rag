"""LLM-as-judge with a fixed rubric (`prompts/judge_v1.md`)."""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic import BaseModel, Field, ValidationError

from docrag.config import PROJECT_ROOT
from docrag.llm.base import LLMClient, LLMResponse

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
            },
        },
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "correctness_reason": {"type": "string"},
    },
    "required": ["claims", "correctness", "correctness_reason"],
}


class Claim(BaseModel):
    claim: str
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    correctness: int = Field(ge=0, le=2)
    correctness_reason: str = ""

    @property
    def n_unsupported(self) -> int:
        return sum(not c.supported for c in self.claims)


class JudgeResult(BaseModel):
    verdict: Verdict | None
    call: LLMResponse | None
    error: str | None = None


@lru_cache(maxsize=2)
def _template(version: str) -> tuple[str, str]:
    text = (PROJECT_ROOT / "prompts" / f"{version}.md").read_text(encoding="utf-8")
    sys_i = text.index("## System\n") + len("## System\n")
    user_i = text.index("## User\n")
    return text[sys_i:user_i].strip(), text[user_i + len("## User\n") :].strip()


def judge_answer(
    llm: LLMClient,
    *,
    question: str,
    expected_behavior: str,
    reference: str,
    context: str,
    answer: str,
    version: str = "judge_v1",
    max_attempts: int = 2,
) -> JudgeResult:
    system, user_t = _template(version)
    prompt = user_t.format(
        question=question,
        expected_behavior=expected_behavior,
        reference=reference or "(none: the documentation does not answer this question)",
        context=context,
        answer=answer,
    )
    last_err = None
    call = None
    for _ in range(max_attempts):
        call = llm.generate(
            prompt, system=system, temperature=0.0, json_schema=JUDGE_SCHEMA, max_output_tokens=2048
        )
        try:
            return JudgeResult(verdict=Verdict.model_validate(json.loads(call.text)), call=call)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = f"{type(e).__name__}: {e}"[:300]
    return JudgeResult(verdict=None, call=call, error=last_err)
