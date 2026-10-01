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


@app.command("eval")
def eval_cmd(
    config: Path | None = ConfigOpt,
    split: str = typer.Option("test", help="test | dev | all"),
    only: str = typer.Option("", help="Comma-separated config names (default: all four)."),
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
        cfg, Settings(), QUESTIONS_PATH, split=split, configs=configs, log=typer.echo
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
