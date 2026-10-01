# CPU-only image: local embeddings + reranker, LLM via API. ~1.5 GB (mostly PyTorch CPU).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/data/hf \
    TQDM_DISABLE=1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first (cached layer); torch comes from the CPU-only index (pyproject.toml).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY prompts ./prompts
RUN uv sync --frozen --no-dev

# Run as an unprivileged user. data/, logs/ and eval/ are bind-mounted volumes
# (docker-compose.yml); corpus and indexes are built into data/ by the `setup` service.
RUN useradd --create-home --uid 1000 app && mkdir -p data logs eval && chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["rag"]
CMD ["bot"]
