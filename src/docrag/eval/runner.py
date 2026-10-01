"""Eval runner: answers every question under every configuration, judges the answers, and
writes one auditable JSON record per (question, configuration).

Reproducibility
- A run directory is named after a hash of everything that affects results (questions,
  corpus commit, chunking, embedding, retrieval and generation settings, prompts, models).
  Re-running the same command resumes that run; any change starts a new run.
- LLM calls go through a persistent cache, so a finished run can be re-scored for free and
  the "citation check off" configuration reuses the first attempts of the "check on" run.
- Free-tier daily quotas: on QuotaExhaustedError the runner stops after saving; re-running
  the same command the next day continues where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from docrag.config import (
    PROJECT_ROOT,
    GenerationConfig,
    PipelineConfig,
    RetrievalConfig,
    RetrievalMode,
    Settings,
)
from docrag.eval.dataset import EvalQuestion, Expected, read_questions, resolve_evidence
from docrag.eval.judge import Verdict, judge_answer
from docrag.generation.assistant import Answer, Assistant
from docrag.generation.prompt import render_sources
from docrag.ingest.pipeline import read_chunks
from docrag.llm import LLMClient, make_client
from docrag.llm.cache import CachedLLM
from docrag.llm.openai_compat import QuotaExhaustedError
from docrag.retrieval.retriever import RetrievedChunk, Retriever

RUNS_DIR = PROJECT_ROOT / "eval" / "runs"


class EvalConfig(BaseModel):
    """One row of the results table."""

    name: str
    mode: RetrievalMode
    citation_check: bool = True


DEFAULT_CONFIGS = [
    EvalConfig(name="baseline_dense", mode=RetrievalMode.DENSE),
    EvalConfig(name="hybrid", mode=RetrievalMode.HYBRID),
    EvalConfig(name="hybrid_rerank", mode=RetrievalMode.HYBRID_RERANK),
    EvalConfig(
        name="hybrid_rerank_no_citecheck", mode=RetrievalMode.HYBRID_RERANK, citation_check=False
    ),
]


class Record(BaseModel):
    qid: str
    config: str
    split: str
    category: str
    subtype: str | None
    expected: str
    question: str
    reference: str | None
    retrieved_ids: list[str]
    top_rerank_score: float | None = None
    gold_groups: list[list[str]]
    answer: str
    abstained: bool
    abstain_reason: str | None
    cited_ids: list[str]
    citation_valid: bool | None
    citation_first_attempt_valid: bool | None
    n_uncited_sentences: int | None = None
    n_llm_calls: int
    input_tokens: int
    output_tokens: int
    retrieval_s: float
    generation_s: float
    verdict: Verdict | None = None
    judge_error: str | None = None
    judge_tokens: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def run_fingerprint(
    cfg: PipelineConfig,
    rcfg: RetrievalConfig,
    gcfg: GenerationConfig,
    settings: Settings,
    questions_path: Path,
    configs: list[EvalConfig],
) -> dict[str, object]:
    prompts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        for p in sorted((PROJECT_ROOT / "prompts").glob("*.md"))
    }
    return {
        "questions_sha": hashlib.sha256(questions_path.read_bytes()).hexdigest()[:12],
        "corpus_commit": cfg.corpus.commit,
        "chunking": cfg.chunking.model_dump(),
        "embedding": cfg.embedding.model_dump(),
        "retrieval": rcfg.model_dump(mode="json"),
        "generation": gcfg.model_dump(mode="json"),
        "provider": settings.llm_provider,
        "generator_model": settings.generator_model,
        "judge_model": settings.judge_model,
        "judge_reasoning_effort": settings.judge_reasoning_effort,
        "prompts": prompts,
        "configs": [c.model_dump(mode="json") for c in configs],
    }


def run_dir_for(fingerprint: dict[str, object]) -> Path:
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:10]
    return RUNS_DIR / f"run-{digest}"


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def _record(
    q: EvalQuestion,
    ec: EvalConfig,
    gold: list[set[str]],
    retrieved: list[RetrievedChunk],
    ans: Answer,
) -> Record:
    first_valid = None
    if ans.llm_calls and ans.citation_check is not None:
        # first attempt is valid iff no retry happened and the final check passed
        first_valid = len(ans.llm_calls) == 1 and ans.citation_check.valid
    return Record(
        qid=q.id,
        config=ec.name,
        split=q.split,
        category=q.category.value,
        subtype=q.subtype,
        expected=q.expected.value,
        question=q.question,
        reference=q.reference_answer,
        retrieved_ids=[r.chunk.chunk_id for r in retrieved],
        top_rerank_score=retrieved[0].rerank_score if retrieved else None,
        gold_groups=[sorted(g) for g in gold],
        answer=ans.text,
        abstained=ans.abstained,
        abstain_reason=ans.abstain_reason.value if ans.abstain_reason else None,
        cited_ids=[c.chunk_id for c in ans.citations],
        citation_valid=ans.citation_check.valid if ans.citation_check else None,
        citation_first_attempt_valid=first_valid,
        n_uncited_sentences=(
            len(ans.citation_check.uncited_sentences) if ans.citation_check else None
        ),
        n_llm_calls=len(ans.llm_calls),
        input_tokens=ans.input_tokens,
        output_tokens=ans.output_tokens,
        retrieval_s=ans.retrieval_s,
        generation_s=sum(c.latency_s for c in ans.llm_calls),
    )


def run_eval(
    cfg: PipelineConfig,
    settings: Settings,
    questions_path: Path,
    split: str = "test",
    configs: list[EvalConfig] | None = None,
    rcfg: RetrievalConfig | None = None,
    gcfg: GenerationConfig | None = None,
    log: Callable[[str], None] = print,
    wait_on_quota: bool = False,
    judge: bool = True,
) -> Path:
    configs = configs or DEFAULT_CONFIGS
    rcfg = rcfg or cfg.retrieval
    gcfg = gcfg or cfg.generation
    fp = run_fingerprint(cfg, rcfg, gcfg, settings, questions_path, configs)
    fp["judged"] = judge  # unjudged runs (e.g. threshold tuning) get their own directory
    run_dir = run_dir_for(fp)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    (run_dir / "fingerprint.json").write_text(json.dumps(fp, indent=2), encoding="utf-8")

    questions = [q for q in read_questions(questions_path) if split == "all" or q.split == split]
    chunks = read_chunks(cfg.chunks_path)
    gold = {q.id: resolve_evidence(q, chunks) for q in questions}
    missing = [q.id for q in questions if q.answerable and q.evidence and not all(gold[q.id])]
    if missing:
        log(f"WARNING: evidence not found in any chunk for {missing}")

    cache = settings.cache_dir / "llm_cache.sqlite"
    gen_llm = CachedLLM(make_client(settings.generator_model, settings), cache)
    judge_llm = (
        CachedLLM(
            make_client(settings.judge_model, settings, settings.judge_reasoning_effort or None),
            cache,
        )
        if judge
        else None
    )
    retriever = Retriever(cfg, rcfg, settings)
    retriever.retrieve("warm-up query", RetrievalMode.HYBRID_RERANK)  # load models before timing
    log(f"Run dir: {run_dir.relative_to(PROJECT_ROOT)}  ({len(questions)} {split} questions)")
    try:
        for ec in configs:
            out = run_dir / "records" / f"{ec.name}.jsonl"
            done = _done(out)
            assistant = Assistant(
                retriever, gen_llm, gcfg.model_copy(update={"citation_check": ec.citation_check})
            )
            todo = [q for q in questions if q.id not in done]
            log(f"[{ec.name}] {len(done)} done, {len(todo)} to go")
            for i, q in enumerate(todo, 1):
                while True:
                    try:
                        rec = _answer_and_judge(q, ec, gold[q.id], assistant, judge_llm)
                        break
                    except QuotaExhaustedError as e:
                        if not wait_on_quota:
                            raise
                        # Completed LLM calls are cached, so retrying the question is free.
                        wait = e.retry_after_s + 30
                        log(
                            f"[{ec.name}] {e.model} daily quota used up; "
                            f"sleeping {wait / 60:.0f} min"
                        )
                        time.sleep(wait)
                with open(out, "a", encoding="utf-8", newline="\n") as f:
                    f.write(rec.model_dump_json() + "\n")
                if i % 10 == 0:
                    log(f"[{ec.name}] {len(done) + i}/{len(questions)}")
    except QuotaExhaustedError as e:
        log(f"STOPPED: {e}. Progress is saved; re-run the same command to resume.")
    finally:
        retriever.close()
        log(
            f"LLM cache: generator {gen_llm.hits} hits / {gen_llm.misses} calls, "
            + (
                f"judge {judge_llm.hits} hits / {judge_llm.misses} calls"
                if judge_llm
                else "no judge"
            )
        )
    return run_dir


def _answer_and_judge(
    q: EvalQuestion,
    ec: EvalConfig,
    gold: list[set[str]],
    assistant: Assistant,
    judge_llm: LLMClient | None,
) -> Record:
    ans = assistant.answer(q.question, ec.mode)
    rec = _record(q, ec, gold, ans.retrieved, ans)
    if judge_llm is not None and not ans.abstained:
        jr = judge_answer(
            judge_llm,
            question=q.question,
            expected_behavior=_expected_text(q.expected),
            reference=q.reference_answer or "",
            context=render_sources(ans.retrieved),
            answer=ans.text,
        )
        rec.verdict = jr.verdict
        rec.judge_error = jr.error
        rec.judge_tokens = (jr.call.input_tokens + jr.call.output_tokens) if jr.call else 0
    return rec


def _expected_text(e: Expected) -> str:
    return {
        Expected.ANSWER: "answer the question",
        Expected.ABSTAIN: "decline: the documentation does not answer this question",
        Expected.CORRECT_PREMISE: "answer while explicitly correcting the question's false premise",
    }[e]
