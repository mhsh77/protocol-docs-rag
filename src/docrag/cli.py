"""Command-line entry point: `rag fetch`, `rag ingest`, `rag eval`, `rag bot`."""

from __future__ import annotations

from pathlib import Path

import typer

from docrag.config import load_pipeline_config
from docrag.corpus.fetch import fetch_corpus, verify_corpus

app = typer.Typer(no_args_is_help=True, add_completion=False)

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
