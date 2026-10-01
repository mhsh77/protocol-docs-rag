"""Helpers for the human review of the drafted question set."""

from __future__ import annotations

import random
from pathlib import Path

from docrag.config import PipelineConfig
from docrag.corpus.fetch import MANIFEST_NAME, read_manifest
from docrag.eval.dataset import EvalQuestion
from docrag.eval.drafting import CONTEXT_DEPENDENT


def lint_question(q: EvalQuestion) -> list[str]:
    """Automatic red flags; every flagged question gets a manual look."""
    issues = []
    if CONTEXT_DEPENDENT.search(q.question):
        issues.append("context-dependent wording")
    if len(q.question.split()) < 5:
        issues.append("very short")
    if q.answerable and not q.evidence and q.subtype != "other_protocol":
        issues.append("no evidence")
    if q.reference_answer and len(q.reference_answer.split()) > 80:
        issues.append("long reference")
    return issues


def render_review_sheet(
    questions: list[EvalQuestion], cfg: PipelineConfig, per_category: int = 5, seed: int = 3
) -> str:
    """A stratified sample as Markdown, with links to the source pages."""
    urls = {e.doc_id: e.published_url for e in read_manifest(cfg.raw_dir / MANIFEST_NAME)}
    rng = random.Random(seed)
    groups: dict[str, list[EvalQuestion]] = {}
    for q in questions:
        groups.setdefault(q.category.value, []).append(q)
    lines = [
        "# Eval question review",
        "",
        "For each question, mark **OK**, **FIX** (say what), or **DROP**. Things to check:",
        "is it a realistic developer question, is the reference answer correct per the linked",
        "page, and for unanswerable ones: are you confident the docs really don't answer it?",
        "",
    ]
    for cat in sorted(groups):
        sample = rng.sample(groups[cat], min(per_category, len(groups[cat])))
        lines += [f"## {cat} ({len(groups[cat])} in set, showing {len(sample)})", ""]
        for q in sample:
            lines.append(
                f"### {q.id} · expected: `{q.expected.value}`"
                + (f" · {q.subtype}" if q.subtype else "")
            )
            lines.append(f"**Q:** {q.question}")
            if q.reference_answer:
                lines.append(f"**Reference:** {q.reference_answer}")
            for ev in q.evidence:
                lines.append(
                    f'- evidence ([source]({urls.get(ev.doc_id, ev.doc_id)})): "{ev.quote}"'
                )
            lines += ["", "Verdict: ", ""]
    return "\n".join(lines)


def write_review_sheet(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
