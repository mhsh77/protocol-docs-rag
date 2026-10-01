"""Chunk-size experiment: pick chunking defaults from data, not a guess.

1. Build a *probe set* (separate from the final eval set): for a seeded sample of docs,
   an LLM writes one question about a random section and copies a verbatim evidence
   quote from it. Quotes are validated as exact substrings of the section.
2. For each chunking config, embed chunks (dense only, cached) and score retrieval:
   hit@k / MRR count a hit when a retrieved chunk contains the evidence quote.
   Because larger chunks trivially contain more quotes, we also report hit@budget:
   a hit within the first N *tokens* of ranked context, i.e. at equal generator cost.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from docrag.config import PipelineConfig
from docrag.corpus.fetch import MANIFEST_NAME, read_manifest
from docrag.ingest.chunking import Chunk, ChunkingConfig, split_sections
from docrag.ingest.mdx import normalize_mdx
from docrag.ingest.pipeline import build_chunks
from docrag.llm.base import LLMClient
from docrag.retrieval.embedder import Embedder


class Probe(BaseModel):
    question: str
    evidence: str
    doc_id: str
    heading_path: list[str]


_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """Whitespace- and markup-insensitive form used for evidence matching."""
    s = s.replace("**", "").replace("`", "")
    return _WS.sub(" ", s).strip().lower()


PROBE_SYSTEM = (
    "You write evaluation questions for a documentation search system. "
    "Questions must sound like a real developer asking, use your own wording rather "
    "than copying phrases from the passage, and be answerable from the passage alone."
)

PROBE_PROMPT = """Documentation section: {heading}

<passage>
{passage}
</passage>

Write ONE specific question answerable from this passage. Then copy an exact evidence
quote from the passage: one contiguous span of 6-30 words that contains the answer,
copied character-for-character (including backticks), from a single sentence or list item."""

PROBE_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "evidence": {"type": "string"}},
    "required": ["question", "evidence"],
}


def build_probe_set(
    cfg: PipelineConfig,
    llm: LLMClient,
    n: int,
    out_path: Path,
    seed: int = 13,
    min_words: int = 60,
) -> list[Probe]:
    """Generate up to `n` probes, appending each to `out_path` as it is made.

    Resumable: docs already present in `out_path` are skipped. Section choice is seeded
    per document, so a resumed run picks the same sections a fresh run would.
    """
    probes = read_probes(out_path) if out_path.exists() else []
    done = {p.doc_id for p in probes}
    entries = read_manifest(cfg.raw_dir / MANIFEST_NAME)
    random.Random(seed).shuffle(entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if len(probes) >= n:
            break
        if entry.doc_id in done:
            continue
        doc = normalize_mdx((cfg.raw_dir / entry.local_path).read_text(encoding="utf-8"))
        sections = [
            s
            for s in split_sections(doc.text, doc.title, lambda t: len(t.split()))
            if sum(b.n_tokens for b in s.blocks if b.kind in ("prose", "list")) >= min_words
        ]
        if not sections:
            continue
        sec = random.Random(f"{seed}:{entry.doc_id}").choice(sections)
        passage = "\n\n".join(b.text for b in sec.blocks)
        passage = " ".join(passage.split(" ")[:600])
        resp = llm.generate(
            PROBE_PROMPT.format(heading=" > ".join(sec.heading_path), passage=passage),
            system=PROBE_SYSTEM,
            temperature=0.7,
            json_schema=PROBE_SCHEMA,
        )
        try:
            data = json.loads(resp.text)
        except json.JSONDecodeError:
            continue
        if norm(data["evidence"]) not in norm(passage) or len(data["evidence"].split()) < 6:
            continue  # reject hallucinated / paraphrased evidence
        probe = Probe(
            question=data["question"],
            evidence=data["evidence"],
            doc_id=entry.doc_id,
            heading_path=sec.heading_path,
        )
        probes.append(probe)
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(probe.model_dump_json() + "\n")
    return probes


def score(
    probes: list[Probe],
    chunks: list[Chunk],
    embedder: Embedder,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    budgets: tuple[int, ...] = (1000, 2000),
) -> ProbeScore:
    mat = embedder.embed_passages([c.embed_text() for c in chunks])
    normed = [norm(c.text) for c in chunks]
    hits = {k: 0 for k in ks}
    bhits = {b: 0 for b in budgets}
    rr = 0.0
    per_probe_rr: list[float] = []
    for p in probes:
        q = embedder.embed_query(p.question)
        order = np.argsort(-(mat @ q))
        ev = norm(p.evidence)
        rank = next((r for r, i in enumerate(order[:50], 1) if ev in normed[i]), None)
        per_probe_rr.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)
        if rank is not None:
            rr += 1.0 / rank if rank <= 10 else 0.0
            for k in ks:
                hits[k] += rank <= k
        for b in budgets:
            used = 0
            for i in order:
                if ev in normed[i]:
                    bhits[b] += 1
                    break
                used += chunks[i].n_tokens
                if used >= b:
                    break
    n = len(probes)
    out = {f"hit@{k}": hits[k] / n for k in ks}
    out["mrr@10"] = rr / n
    out.update({f"hit@{b}tok": bhits[b] / n for b in budgets})
    out["n_chunks"] = float(len(chunks))
    out["median_tokens"] = float(np.median([c.n_tokens for c in chunks]))
    # Mean tokens of the top-5 chunks actually retrieved: what the generator pays per question.
    out["top5_tokens"] = float(
        np.mean(
            [
                sum(
                    chunks[i].n_tokens
                    for i in np.argsort(-(mat @ embedder.embed_query(p.question)))[:5]
                )
                for p in probes
            ]
        )
    )
    return ProbeScore(metrics=out, reciprocal_ranks=per_probe_rr)


class ProbeScore(BaseModel):
    metrics: dict[str, float]
    reciprocal_ranks: list[float]  # per probe, aligned with the probe file, for paired CIs


def run_grid(
    cfg: PipelineConfig, probes: list[Probe], grid: list[ChunkingConfig], embedder: Embedder
) -> list[tuple[ChunkingConfig, ProbeScore]]:
    return [(cc, score(probes, build_chunks(cfg, cc), embedder)) for cc in grid]


def write_probes(probes: list[Probe], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in probes:
            f.write(p.model_dump_json() + "\n")


def read_probes(path: Path) -> list[Probe]:
    with open(path, encoding="utf-8") as f:
        return [Probe.model_validate_json(line) for line in f if line.strip()]


DEFAULT_GRID = [
    ChunkingConfig(max_tokens=size, overlap_tokens=ov, min_section_tokens=0)
    for size in (128, 256, 512, 1024)
    for ov in (0, size // 8)
]


def render_markdown(
    results: list[tuple[ChunkingConfig, ProbeScore]], n_probes: int, model: str
) -> str:
    cols = ["hit@1", "hit@5", "mrr@10", "hit@2000tok", "n_chunks", "median_tokens", "top5_tokens"]
    lines = [
        "| max_tokens | overlap | merge<N | " + " | ".join(cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for cc, s in results:
        m = s.metrics
        vals = [f"{m[c]:.0f}" if c.endswith(("chunks", "tokens")) else f"{m[c]:.3f}" for c in cols]
        lines.append(
            f"| {cc.max_tokens} | {cc.overlap_tokens} | {cc.min_section_tokens} | "
            + " | ".join(vals)
            + " |"
        )
    header = (
        f"Dense-only retrieval with `{model}`, {n_probes} probe questions "
        "(`eval/probe/chunk_probe.jsonl`). A hit = a retrieved chunk contains the "
        "probe's verbatim evidence quote. `hit@Ntok` = hit within the first N tokens of "
        "ranked context (equal generator cost across chunk sizes).\n\n"
    )
    return header + "\n".join(lines) + "\n"
