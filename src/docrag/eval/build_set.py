"""Orchestrates drafting of the eval set (resumable; appends to a drafts file).

Drafting uses the *judge* model (not the generator), so the system under test is not
answering questions it wrote itself. Unanswerable candidates are verified with a third
model to avoid one model's blind spots deciding what "unanswerable" means.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml

from docrag.config import PipelineConfig, RetrievalMode
from docrag.corpus.fetch import MANIFEST_NAME, read_manifest
from docrag.eval.dataset import (
    Category,
    EvalQuestion,
    Expected,
    append_question,
    read_questions,
    write_questions,
)
from docrag.eval.drafting import (
    OTHER_PROTOCOL_QUESTIONS,
    Section,
    draft_multi,
    draft_single,
    draft_unanswerable,
    has_numeric,
    iter_sections,
    make_question,
    verify_unanswerable,
)
from docrag.llm.base import LLMClient
from docrag.retrieval.retriever import Retriever

# (category, subtype) -> target count. Unanswerable >= 20 as required, plus margin.
TARGETS: dict[tuple[Category, str | None], int] = {
    (Category.FACTUAL, None): 25,
    (Category.NUMERIC, None): 20,
    (Category.MULTI_SECTION, None): 20,
    (Category.UNANSWERABLE, None): 28,
    (Category.ADVERSARIAL, "false_premise"): 8,
    # LLM drafts of this subtype were mostly invalid (4 of 6 dropped in review); the set uses
    # 2 curated drafts + 3 hand-written questions with verified evidence (eval/curation.yaml).
    (Category.ADVERSARIAL, "outdated_term"): 5,
    (Category.ADVERSARIAL, "other_protocol"): 7,
}
DEV_FRACTION = 0.27  # ~30 of ~114 questions for tuning; the rest is the held-out test split


def _key(sec: Section) -> str:
    return f"{sec.entry.doc_id} :: {' > '.join(sec.heading_path)}"


def build_drafts(
    cfg: PipelineConfig,
    drafter: LLMClient,
    verifier: LLMClient,
    retriever: Retriever,
    out_path: Path,
    curation_path: Path | None = None,
    seed: int = 7,
    log: Callable[[str], None] = print,
) -> list[EvalQuestion]:
    existing = read_questions(out_path) if out_path.exists() else []
    curated = apply_curation(existing, load_curation(curation_path)) if curation_path else existing
    counts = Counter((q.category, q.subtype) for q in curated)
    used = {q.notes for q in existing if q.notes}
    rng = random.Random(seed)
    sections = list(iter_sections(cfg))
    rng.shuffle(sections)
    fresh = [s for s in sections if _key(s) not in used]
    entries = {e.doc_id: e for e in read_manifest(cfg.raw_dir / MANIFEST_NAME)}
    n = len(existing)

    def add(q: EvalQuestion) -> None:
        nonlocal n
        append_question(q, out_path)
        counts[(q.category, q.subtype)] += 1
        if q.notes:
            used.add(q.notes)
        n += 1
        log(f"[{n}] {q.category}/{q.subtype or '-'}: {q.question[:90]}")

    def need(cat: Category, sub: str | None = None) -> bool:
        return counts[(cat, sub)] < TARGETS[(cat, sub)]

    def next_section(pred: Callable[[Section], bool] = lambda s: True) -> Section | None:
        while fresh:
            s = fresh.pop()
            if _key(s) not in used and pred(s):
                return s
        return None

    # Hand-written other-protocol questions (no LLM).
    for i, text in enumerate(OTHER_PROTOCOL_QUESTIONS):
        if need(Category.ADVERSARIAL, "other_protocol"):
            add(
                make_question(
                    f"draft-op-{i}",
                    "",
                    Category.ADVERSARIAL,
                    text,
                    Expected.ABSTAIN,
                    subtype="other_protocol",
                    source="manual",
                    notes=f"manual-other-protocol-{i}",
                )
            )

    single_kinds: list[tuple[Category, str | None, Expected, Callable[[Section], bool]]] = [
        (Category.NUMERIC, None, Expected.ANSWER, has_numeric),
        (Category.FACTUAL, None, Expected.ANSWER, lambda s: True),
        (Category.ADVERSARIAL, "false_premise", Expected.CORRECT_PREMISE, lambda s: True),
        (Category.ADVERSARIAL, "outdated_term", Expected.ANSWER, lambda s: True),
    ]
    for cat, sub, expected, pred in single_kinds:
        tries = 0
        while need(cat, sub) and tries < TARGETS[(cat, sub)] * 3:
            tries += 1
            sec = next_section(pred)
            if sec is None:
                break
            out = draft_single(drafter, sub or cat, sec)
            if out is None:
                used.add(_key(sec))
                continue
            question, ref, ev = out
            add(
                make_question(
                    f"draft-{n}", "", cat, question, expected, ref, ev, sub, notes=_key(sec)
                )
            )

    tries = 0
    while need(Category.MULTI_SECTION) and tries < TARGETS[(Category.MULTI_SECTION, None)] * 3:
        tries += 1
        a = next_section()
        if a is None:
            break
        query = f"{' > '.join(a.heading_path)}. {a.text[:300]}"
        hits = retriever.retrieve(query, RetrievalMode.HYBRID)
        other = next((h.chunk for h in hits if h.chunk.doc_id != a.entry.doc_id), None)
        if other is None:
            continue
        b = Section(entries[other.doc_id], other.heading_path, other.text)
        out = draft_multi(drafter, a, b)
        used.add(_key(a))
        if out is None:
            continue
        question, ref, ev = out
        add(
            make_question(
                f"draft-{n}",
                "",
                Category.MULTI_SECTION,
                question,
                Expected.ANSWER,
                ref,
                ev,
                notes=f"{_key(a)} || {other.doc_id} :: {other.heading_str}",
            )
        )

    # Unanswerable: brainstorm per top-level docs area, verify each candidate.
    areas: dict[str, list[str]] = {}
    for s in sections:
        area = s.entry.local_path.split("/")[0]
        areas.setdefault(area, []).append(" > ".join(s.heading_path))
    area_names = sorted(areas)
    rng.shuffle(area_names)
    rounds = 0
    while need(Category.UNANSWERABLE) and rounds < 3 * len(area_names):
        area = area_names[rounds % len(area_names)]
        rounds += 1
        for cand in draft_unanswerable(
            drafter, rng.sample(areas[area], min(25, len(areas[area]))), 4
        ):
            if not need(Category.UNANSWERABLE) or f"unans: {cand}" in used:
                continue
            ok, reason = verify_unanswerable(verifier, retriever, cand)
            if ok:
                add(
                    make_question(
                        f"draft-{n}",
                        "",
                        Category.UNANSWERABLE,
                        cand,
                        Expected.ABSTAIN,
                        notes=f"unans: {cand}",
                    )
                )
            else:
                used.add(f"unans: {cand}")
                log(f"   rejected (docs answer it: {reason[:80]}): {cand[:70]}")

    return read_questions(out_path)


def assign_splits(questions: list[EvalQuestion], seed: int = 11) -> list[EvalQuestion]:
    """Stratified dev/test split per (category, subtype); ids become stable `q###`."""
    rng = random.Random(seed)
    groups: dict[tuple[Category, str | None], list[EvalQuestion]] = {}
    for q in questions:
        groups.setdefault((q.category, q.subtype), []).append(q)
    out: list[EvalQuestion] = []
    for key in sorted(groups, key=lambda k: (k[0].value, k[1] or "")):
        g = groups[key]
        rng.shuffle(g)
        n_dev = round(len(g) * DEV_FRACTION)
        for i, q in enumerate(g):
            split = "dev" if i < n_dev else "test"
            out.append(q.model_copy(update={"split": split, "draft_id": q.draft_id or q.id}))
    for i, q in enumerate(out, 1):
        q.id = f"q{i:03d}"
    return out


def load_curation(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: (data.get(k) or {}) for k in ("drop", "rewrite", "recategorize")}


def apply_curation(
    questions: list[EvalQuestion], curation: dict[str, dict[str, object]]
) -> list[EvalQuestion]:
    """Apply human review decisions. Unknown ids in the curation file are an error."""
    ids = {q.id for q in questions}
    for section in curation.values():
        unknown = set(section) - ids
        if unknown:
            raise ValueError(f"curation refers to unknown draft ids: {sorted(unknown)}")
    out = []
    for q in questions:
        if q.id in curation.get("drop", {}):
            continue
        update: dict[str, object] = {}
        rw = curation.get("rewrite", {}).get(q.id)
        if isinstance(rw, dict):
            if "question" in rw:
                update["question"] = rw["question"]
            if "reference" in rw:
                update["reference_answer"] = rw["reference"]
            update["source"] = "llm_draft+human_edit"
        cat = curation.get("recategorize", {}).get(q.id)
        if cat:
            update["category"] = Category(str(cat))
        out.append(q.model_copy(update=update) if update else q)
    return out


def finalize(drafts_path: Path, out_path: Path, curation_path: Path) -> list[EvalQuestion]:
    qs = apply_curation(read_questions(drafts_path), load_curation(curation_path))
    qs = assign_splits(qs)
    write_questions(qs, out_path)
    return qs
