"""Prompt template loading and source rendering.

The template text lives in `prompts/answer_v1.md` (single source of truth, documented
in the repo); this module only parses it and fills it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from docrag.config import PROJECT_ROOT
from docrag.retrieval.retriever import RetrievedChunk

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "answer": {"type": "string"},
        "closest_source": {"type": ["string", "null"]},
    },
    "required": ["abstain", "answer", "closest_source"],
}


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str


@lru_cache(maxsize=4)
def load_template(version: str = "answer_v1") -> PromptTemplate:
    text = (PROJECT_ROOT / "prompts" / f"{version}.md").read_text(encoding="utf-8")
    system = _section(text, "## System\n", "## User\n")
    user = _section(text, "## User\n", "\nEach source block")
    return PromptTemplate(version=version, system=system, user=user)


def _section(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    return text[i : text.index(end, i)].strip()


def alias(i: int) -> str:
    return f"S{i + 1}"


def render_sources(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(retrieved):
        c = r.chunk
        blocks.append(f"[{alias(i)}] {c.heading_str}\nURL: {c.url}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(
    question: str, retrieved: list[RetrievedChunk], protocol_name: str, template: PromptTemplate
) -> tuple[str, str]:
    system = template.system.format(protocol_name=protocol_name)
    user = template.user.format(question=question.strip(), sources=render_sources(retrieved))
    return system, user
