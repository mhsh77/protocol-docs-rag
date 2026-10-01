"""Command-line entry point: `rag fetch`, `rag ingest`, `rag eval`, `rag bot`."""

from __future__ import annotations

from pathlib import Path

import typer

from docrag.config import PROJECT_ROOT, Settings, load_pipeline_config
from docrag.corpus.fetch import fetch_corpus, verify_corpus
from docrag.ingest.pipeline import build_chunks, write_chunks
from docrag.retrieval.retriever import build_index, index_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)

DRAFTS_PATH = PROJECT_ROOT / "eval" / "drafts" / "questions_draft.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
CURATION_PATH = PROJECT_ROOT / "eval" / "curation.yaml"

ConfigOpt = typer.Option(None, "--config", "-c", help="Pipeline YAML (default: $CORPUS_CONFIG).")


@app.command()
def fetch(config: Path | None = ConfigOpt) -> None:
    """Download the pinned docs snapshot into data/raw/<corpus>/ and write the manifest."""
    cfg = load_pipeline_config(config)
    entries = fetch_corpus(cfg.corpus, cfg.raw_dir)
    total_kb = sum(e.bytes for e in entries) / 1024
    typer.echo(f"Fetched {len(entries)} files ({total_kb:.0f} KB) @ {cfg.corpus.commit[:7]}")
    _report_verify(cfg.raw_dir)


@app.command()
def ingest(config: Path | None = ConfigOpt) -> None:
    """Normalize and chunk the raw corpus into data/processed/<corpus>/chunks.jsonl."""
    cfg = load_pipeline_config(config)
    _report_verify(cfg.raw_dir)
    chunks = build_chunks(cfg)
    write_chunks(chunks, cfg.chunks_path)
    sizes = sorted(c.n_tokens for c in chunks)
    typer.echo(
        f"Wrote {len(chunks)} chunks -> {cfg.chunks_path.relative_to(PROJECT_ROOT)} "
        f"(tokens p50={sizes[len(sizes) // 2]}, p95={sizes[int(len(sizes) * 0.95)]}, "
        f"max={sizes[-1]})"
    )
    typer.echo("Embedding (cached) and building dense + BM25 indexes ...")
    n = build_index(cfg)
    typer.echo(f"Indexed {n} chunks -> {index_dir(cfg).relative_to(PROJECT_ROOT)}")


@app.command("draft-questions")
def draft_questions(
    config: Path | None = ConfigOpt,
    verifier_model: str = typer.Option("openai/gpt-oss-20b", help="Model for unanswerable checks."),
) -> None:
    """Draft eval questions (resumable) into eval/drafts/questions_draft.jsonl."""
    from docrag.config import RetrievalConfig
    from docrag.eval.build_set import build_drafts
    from docrag.llm import make_client
    from docrag.retrieval.retriever import Retriever

    cfg = load_pipeline_config(config)
    settings = Settings()
    retriever = Retriever(cfg, RetrievalConfig(), settings)
    try:
        qs = build_drafts(
            cfg,
            drafter=make_client(settings.judge_model, settings),
            verifier=make_client(verifier_model, settings),
            retriever=retriever,
            out_path=DRAFTS_PATH,
            curation_path=CURATION_PATH,
            log=typer.echo,
        )
    finally:
        retriever.close()
    typer.echo(f"{len(qs)} drafted questions in {DRAFTS_PATH.relative_to(PROJECT_ROOT)}")


@app.command("finalize-questions")
def finalize_questions() -> None:
    """Assign stratified dev/test splits and stable ids -> eval/questions.jsonl."""
    from docrag.eval.build_set import finalize

    qs = finalize(DRAFTS_PATH, QUESTIONS_PATH, CURATION_PATH)
    dev = sum(q.split == "dev" for q in qs)
    typer.echo(f"{len(qs)} questions ({dev} dev / {len(qs) - dev} test) -> {QUESTIONS_PATH.name}")


@app.command("review-sheet")
def review_sheet(
    config: Path | None = ConfigOpt,
    per_category: int = typer.Option(5, help="Questions sampled per category."),
) -> None:
    """Write a stratified sample of eval/questions.jsonl for human review."""
    from docrag.eval.dataset import read_questions
    from docrag.eval.review import lint_question, render_review_sheet, write_review_sheet

    cfg = load_pipeline_config(config)
    qs = read_questions(QUESTIONS_PATH)
    flagged = [(q.id, issues) for q in qs if (issues := lint_question(q))]
    for qid, issues in flagged:
        typer.echo(f"  lint {qid}: {', '.join(issues)}")
    out = PROJECT_ROOT / "eval" / "review" / "review_sample.md"
    write_review_sheet(render_review_sheet(qs, cfg, per_category), out)
    typer.echo(f"{len(flagged)} lint flags; review sheet -> {out.relative_to(PROJECT_ROOT)}")


@app.command("judge-sheet")
def judge_sheet(
    run: Path = typer.Argument(..., help="Run directory to sample judged answers from."),
    n: int = typer.Option(20, help="Number of items."),
) -> None:
    """Write a 20-item judge spot-check sheet (no domain knowledge needed)."""
    from docrag.eval.judge_check import render_sheet, sample_for_review
    from docrag.eval.report import load_records

    recs = [r for rs in load_records(run).values() for r in rs]
    out = run / "judge_spot_check.md"
    out.write_text(render_sheet(sample_for_review(recs, n)), encoding="utf-8")
    typer.echo(f"Sheet -> {out}")


@app.command("judge-agreement")
def judge_agreement(
    run: Path = typer.Argument(..., help="Run directory with a filled sheet."),
) -> None:
    """Report agreement between the filled spot-check sheet and the LLM judge."""
    from docrag.eval.judge_check import agreement
    from docrag.eval.report import load_records

    recs = [r for rs in load_records(run).values() for r in rs]
    res = agreement(run / "judge_spot_check.md", recs)
    (run / "judge_agreement.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"{res.n} items: agreement {res.agreement:.0%}, Cohen's kappa {res.kappa:.2f}")
    for item, human, judge in res.disagreements:
        typer.echo(f"  {item}: reviewer {human}, judge {judge}")


@app.command("tune-threshold")
def tune_threshold(
    run: Path = typer.Argument(..., help="Dev-split run directory with hybrid_rerank records."),
    max_false_abstention: float = typer.Option(0.10, help="Cap on wrongly declined answerables."),
) -> None:
    """Pick the reranker-score abstention gate on the DEV split and document the sweep."""
    from docrag.eval.report import load_records
    from docrag.eval.threshold import best_threshold, sweep

    recs = [r for r in load_records(run).get("hybrid_rerank", []) if r.split == "dev"]
    if not recs:
        raise typer.BadParameter("no dev-split hybrid_rerank records in that run")
    best = best_threshold(recs, max_false_abstention)
    points = sorted(
        sweep(recs, max_false_abstention), key=lambda p: (p.threshold is not None, p.threshold or 0)
    )
    lines = [
        "# Abstention threshold (retrieval-score gate)",
        "",
        f"Tuned on the **dev split only** ({len(recs)} questions, run `{run.name}`): "
        "abstain without",
        "calling the LLM when the top reranker score is below the threshold. Objective: maximise",
        "correct abstention minus false abstention, with false abstention <= "
        f"{max_false_abstention:.0%}.",
        "",
        "| Threshold | Correct abstention | False abstention |",
        "|---|---|---|",
    ]
    seen: set[tuple[float, float]] = set()
    for p in points:
        key = (round(p.correct_abstention, 4), round(p.false_abstention, 4))
        if key in seen and p.threshold is not None:
            continue
        seen.add(key)
        t = "none (model only)" if p.threshold is None else f"{p.threshold:.3f}"
        lines.append(f"| {t} | {p.correct_abstention:.1%} | {p.false_abstention:.1%} |")
    chosen = "none" if best.threshold is None else f"{best.threshold:.3f}"
    lines += [
        "",
        f"**Chosen: {chosen}** (correct abstention {best.correct_abstention:.1%}, "
        f"false abstention {best.false_abstention:.1%} on dev).",
        "",
        "Set it as `generation.min_rerank_score` in `config/uniswap.yaml`. With few dev",
        "questions this is a coarse estimate; the test split measures it out of sample.",
    ]
    out = PROJECT_ROOT / "docs" / "abstention-threshold.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo("\n".join(lines))


@app.command("eval")
def eval_cmd(
    config: Path | None = ConfigOpt,
    split: str = typer.Option("test", help="test | dev | all"),
    only: str = typer.Option("", help="Comma-separated config names (default: all four)."),
    wait_on_quota: bool = typer.Option(
        False, help="On a daily-quota stop, sleep until the provider resets and continue."
    ),
    no_judge: bool = typer.Option(
        False, help="Skip LLM judging (enough for retrieval metrics and threshold tuning)."
    ),
) -> None:
    """Run the full evaluation (resumable) and write the results table."""
    from docrag.eval.report import write_report
    from docrag.eval.runner import DEFAULT_CONFIGS, run_eval

    cfg = load_pipeline_config(config)
    configs = DEFAULT_CONFIGS
    if only:
        names = {n.strip() for n in only.split(",")}
        configs = [c for c in DEFAULT_CONFIGS if c.name in names]
    run_dir = run_eval(
        cfg,
        Settings(),
        QUESTIONS_PATH,
        split=split,
        configs=configs,
        log=typer.echo,
        wait_on_quota=wait_on_quota,
        judge=not no_judge,
    )
    write_report(run_dir, split)
    typer.echo((run_dir / "results.md").read_text(encoding="utf-8"))


@app.command()
def report(
    run: Path = typer.Argument(..., help="Run directory, e.g. eval/runs/run-abc123"),
    split: str = typer.Option("test"),
    update_readme: bool = typer.Option(False, help="Replace the README results block."),
) -> None:
    """Recompute the results table from a run's saved records (no LLM calls)."""
    from docrag.eval.report import update_readme_block, write_report

    write_report(run, split)
    table = (run / "results.md").read_text(encoding="utf-8")
    typer.echo(table)
    if update_readme:
        update_readme_block(PROJECT_ROOT / "README.md", table)
        typer.echo("README results block updated.")


@app.command()
def bot(config: Path | None = ConfigOpt) -> None:
    """Start the Telegram bot (long polling)."""
    from docrag.bot.telegram_bot import run
    from docrag.service import build_service

    cfg = load_pipeline_config(config)
    settings = Settings()
    if not settings.telegram_bot_token:
        raise typer.BadParameter("TELEGRAM_BOT_TOKEN is not set (see .env.example).")
    run(settings.telegram_bot_token, build_service(cfg, settings), cfg.generation.protocol_name)


@app.command()
def verify(config: Path | None = ConfigOpt) -> None:
    """Check the raw corpus against its manifest (presence + sha256)."""
    cfg = load_pipeline_config(config)
    _report_verify(cfg.raw_dir)


def _report_verify(raw_dir: Path) -> None:
    problems = verify_corpus(raw_dir)
    if problems:
        for p in problems:
            typer.echo(f"  {p}", err=True)
        raise typer.Exit(code=1)
    typer.echo("Corpus verified against manifest.")


if __name__ == "__main__":
    app()
