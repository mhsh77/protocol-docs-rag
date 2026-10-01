"""Draft eval questions from the corpus with an LLM, with automatic validation.

Every drafted answerable question must carry verbatim evidence quotes that (a) occur in
the source section and (b) resolve to at least one chunk. Unanswerable questions are
kept only if a retrieval + LLM check finds no answer in the docs. Everything drafted
here is then human-reviewed before the set is treated as final.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass

from docrag.config import PipelineConfig, RetrievalMode
from docrag.corpus.fetch import MANIFEST_NAME, ManifestEntry, read_manifest
from docrag.eval.dataset import Category, EvalQuestion, Evidence, Expected, norm
from docrag.ingest.chunking import split_sections
from docrag.ingest.mdx import normalize_mdx
from docrag.llm.base import LLMClient
from docrag.retrieval.retriever import Retriever

_WORDS = lambda t: len(t.split())  # noqa: E731  (token proxy for section sizing)
_NUMERIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)*\s*(?:%|bps|basis points|seconds|blocks|days|wei|gwei|bits?)\b"
    r"|\b0x[0-9a-fA-F]{8,}\b|\b\d{2,}\b"
)


@dataclass
class Section:
    entry: ManifestEntry
    heading_path: list[str]
    text: str


def iter_sections(cfg: PipelineConfig, min_words: int = 60) -> Iterator[Section]:
    for entry in read_manifest(cfg.raw_dir / MANIFEST_NAME):
        doc = normalize_mdx((cfg.raw_dir / entry.local_path).read_text(encoding="utf-8"))
        for s in split_sections(doc.text, doc.title, _WORDS):
            text = "\n\n".join(b.text for b in s.blocks)
            prose = sum(b.n_tokens for b in s.blocks if b.kind in ("prose", "list", "table"))
            if prose >= min_words:
                yield Section(entry, s.heading_path, " ".join(text.split(" ")[:700]))


SYSTEM = (
    "You write evaluation questions for a documentation Q&A assistant about the Uniswap "
    "developer docs. Questions must sound like a real developer asking, in your own words "
    "(do not copy long phrases from the passage). Evidence quotes must be copied exactly, "
    "character for character, from the passage."
)

_ANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "reference_answer", "evidence"],
}

PROMPTS = {
    Category.FACTUAL: """Section: {heading}
<passage>
{passage}
</passage>
Write ONE factual question answerable from this single passage. Give a concise reference
answer (1-3 sentences) and 1-2 exact evidence quotes (each 6-40 words, one sentence or list
item or table row) that support the answer.""",
    Category.NUMERIC: """Section: {heading}
<passage>
{passage}
</passage>
Write ONE question whose answer is an exact value stated in this passage (a number, fee,
percentage, address, parameter value, limit, or constant). The reference answer must
state the exact value. Give 1 exact evidence quote (6-40 words) containing the value.""",
    "false_premise": """Section: {heading}
<passage>
{passage}
</passage>
Write ONE question that contains a FALSE PREMISE contradicted by this passage (for example
asking "why" something is true when the passage says the opposite, or using a wrong
number). The reference answer must correct the premise using the passage. Give 1 exact
evidence quote (6-40 words) that shows the premise is false.""",
    "outdated_term": """Section: {heading}
<passage>
{passage}
</passage>
Write ONE question about this passage phrased with outdated, legacy or informal terminology
a developer might use (e.g. an older Uniswap version's name for a concept, or an imprecise
community term), while the answer is still in the passage. The reference answer uses the
current terminology. Give 1 exact evidence quote (6-40 words).""",
}

MULTI_PROMPT = """Two documentation sections:

<passage_a section="{heading_a}">
{passage_a}
</passage_a>

<passage_b section="{heading_b}">
{passage_b}
</passage_b>

If these sections are related, write ONE question that can only be answered fully by
combining information from BOTH passages. Give a concise reference answer and exactly two
evidence quotes: the first copied from passage A, the second from passage B (each 6-40
words). If they are unrelated, return an empty question string."""


def _validate(
    data: dict[str, object], passages: list[tuple[ManifestEntry, str]]
) -> list[Evidence] | None:
    """Map each quote to the passage containing it; reject on any miss."""
    quotes = data.get("evidence")
    if not isinstance(quotes, list) or not quotes or not str(data.get("question", "")).strip():
        return None
    out = []
    for q in quotes:
        q = str(q)
        if len(q.split()) < 4:
            return None
        hit = next((e for e, text in passages if norm(q) in norm(text)), None)
        if hit is None:
            return None
        out.append(Evidence(doc_id=hit.doc_id, quote=q))
    return out


def _call(llm: LLMClient, prompt: str) -> dict[str, object] | None:
    resp = llm.generate(
        prompt,
        system=SYSTEM,
        temperature=0.7,
        json_schema=_ANSWERABLE_SCHEMA,
        max_output_tokens=3000,
    )
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def draft_single(
    llm: LLMClient, kind: Category | str, sec: Section
) -> tuple[str, str, list[Evidence]] | None:
    heading = " > ".join(sec.heading_path)
    data = _call(llm, PROMPTS[kind].format(heading=heading, passage=sec.text))
    if data is None:
        return None
    ev = _validate(data, [(sec.entry, sec.text)])
    if ev is None:
        return None
    return str(data["question"]), str(data["reference_answer"]), ev


def draft_multi(llm: LLMClient, a: Section, b: Section) -> tuple[str, str, list[Evidence]] | None:
    prompt = MULTI_PROMPT.format(
        heading_a=" > ".join(a.heading_path),
        passage_a=a.text,
        heading_b=" > ".join(b.heading_path),
        passage_b=b.text,
    )
    data = _call(llm, prompt)
    if data is None:
        return None
    ev = _validate(data, [(a.entry, a.text), (b.entry, b.text)])
    if ev is None or len(ev) < 2 or len({(e.doc_id, e.quote) for e in ev}) < 2:
        return None
    if not (norm(ev[0].quote) in norm(a.text) and norm(ev[1].quote) in norm(b.text)):
        return None
    return str(data["question"]), str(data["reference_answer"]), ev


UNANSWERABLE_PROMPT = """Here are the titles of sections in the Uniswap developer documentation
on one topic:
{headings}

Write {n} questions a developer might plausibly ask about this topic that the documentation
very likely does NOT answer: specific details, numbers, timelines, comparisons, internal
policies, or behaviour that documentation of this kind normally omits. They must sound
realistic and on-topic, not absurd. Return JSON: {{"questions": ["...", "..."]}}"""

UNANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}

VERIFY_PROMPT = """Question: {question}

Documentation excerpts:
{context}

Do these excerpts answer the question, fully or partially? Be strict: "partial" if any
key part of the answer is present. Return JSON: {{"answered": "yes" | "partial" | "no",
"reason": "one sentence"}}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "string", "enum": ["yes", "partial", "no"]},
        "reason": {"type": "string"},
    },
    "required": ["answered", "reason"],
}


def draft_unanswerable(llm: LLMClient, headings: list[str], n: int) -> list[str]:
    resp = llm.generate(
        UNANSWERABLE_PROMPT.format(headings="\n".join(f"- {h}" for h in headings[:25]), n=n),
        system=SYSTEM,
        temperature=0.8,
        json_schema=UNANSWERABLE_SCHEMA,
        max_output_tokens=3000,
    )
    try:
        qs = json.loads(resp.text).get("questions", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [str(q).strip() for q in qs if str(q).strip()]


def verify_unanswerable(
    llm: LLMClient, retriever: Retriever, question: str, k: int = 8
) -> tuple[bool, str]:
    """True if neither hybrid retrieval top-k nor the verifier finds an answer."""
    hits = retriever.retrieve(question, RetrievalMode.HYBRID)[:k]
    context = "\n\n---\n\n".join(f"[{h.chunk.heading_str}]\n{h.chunk.text}" for h in hits)
    resp = llm.generate(
        VERIFY_PROMPT.format(question=question, context=context),
        temperature=0.0,
        json_schema=VERIFY_SCHEMA,
        max_output_tokens=800,
    )
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        return False, "verifier output not JSON"
    return data.get("answered") == "no", str(data.get("reason", ""))


# Hand-written: questions about other protocols' mechanics, phrased as if about Uniswap.
OTHER_PROTOCOL_QUESTIONS = [
    "What health factor triggers liquidation of a borrow position in Uniswap v3?",
    "How do I stake UNI in the Uniswap safety module to earn staking rewards?",
    "What is the amplification coefficient used by Uniswap v4 stable pools?",
    "How do I mint DAI against my Uniswap LP position as collateral?",
    "What is the borrow APY for USDC on Uniswap v4 lending markets?",
    "How do veUNI vote-escrowed locks boost my liquidity mining rewards on Uniswap?",
    "What is the maximum leverage for perpetual futures trading on Uniswap?",
]


def pick(rng: random.Random, items: list[Section], n: int) -> list[Section]:
    return rng.sample(items, min(n, len(items)))


def make_question(
    qid: str,
    split: str,
    category: Category,
    question: str,
    expected: Expected,
    reference: str | None = None,
    evidence: list[Evidence] | None = None,
    subtype: str | None = None,
    source: str = "llm_draft",
    notes: str | None = None,
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        split=split,
        category=category,
        subtype=subtype,
        question=question,
        expected=expected,
        reference_answer=reference,
        evidence=evidence or [],
        source=source,
        notes=notes,
    )


def has_numeric(sec: Section) -> bool:
    return len(_NUMERIC_RE.findall(sec.text)) >= 2
