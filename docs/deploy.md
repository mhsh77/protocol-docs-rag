# Deploying on a small VPS

Everything except the LLM runs on the server's CPU (embeddings, BM25, reranker, Qdrant in
embedded mode), so no GPU and no database service are needed. Two ways to deploy:

- **systemd, no Docker** (what the live demo uses): lightest option, for a small shared box.
- **Docker Compose**: self-contained, for a dedicated server with 2+ GB RAM (section below).

## Option A: systemd on a small shared VPS (live demo setup)

The live bot runs this way on a 1 vCPU / 1 GB RAM Debian 13 VPS that also hosts other
services. Measured there: peak ~680 MB RSS, retrieval 2.5–4.3 s per question.
The unit file [`deploy/docrag-bot.service`](../deploy/docrag-bot.service) makes the bot the
first process the kernel kills under memory pressure (`OOMScoreAdjust=800`), caps it at
1.1 GB (`MemoryMax`), and gives it low CPU priority, so other services on the host win.

```bash
# on the server, as root
useradd --system --create-home --home-dir /opt/docrag --shell /usr/sbin/nologin docrag
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
# copy the code to /opt/docrag (git clone, or `git archive HEAD | ssh ... tar -x`)
# copy a prebuilt index from your laptop to skip embedding on a small CPU:
#   data/raw/<corpus>, data/processed/<corpus>, data/index/<corpus>
# create /opt/docrag/.env (chmod 600, owned by docrag) with GROQ_API_KEY,
#   TELEGRAM_BOT_TOKEN, GENERATOR_MODEL, LLM_PROVIDER=groq, LOG_SALT
chown -R docrag:docrag /opt/docrag
cd /opt/docrag && sudo -u docrag env HOME=/opt/docrag uv sync --frozen --no-dev
cp deploy/docrag-bot.service /etc/systemd/system/
mkdir -p /opt/docrag/logs && chown docrag:docrag /opt/docrag/logs
systemctl daemon-reload && systemctl enable --now docrag-bot
journalctl -u docrag-bot -f      # look for "Application started"
```

Update: copy the new code, `uv sync --frozen --no-dev`, `systemctl restart docrag-bot`.

## Option B: Docker Compose

## Sizing

| Resource | Minimum | Notes |
|---|---|---|
| RAM | 2 GB (4 GB recommended) | embedding model + reranker + PyTorch ≈ 1.5 GB resident |
| Disk | 6 GB | image ≈ 1.5 GB, HF models ≈ 1.5 GB, indexes + caches < 200 MB |
| CPU | 2 vCPU | the reranker dominates latency; see the latency column in the results table |

### 1. Install Docker (Ubuntu/Debian)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
```

### 2. Get the code and configure secrets

```bash
git clone <your-repo-url> docrag && cd docrag
cp .env.example .env
nano .env            # set GROQ_API_KEY, TELEGRAM_BOT_TOKEN, GENERATOR_MODEL, JUDGE_MODEL, LOG_SALT
chmod 600 .env
mkdir -p data logs eval && sudo chown -R 1000:1000 data logs eval   # container runs as uid 1000
```

### 3. Build and index (one-off)

```bash
docker compose build
docker compose run --rm setup
```

`setup` downloads the pinned docs snapshot, verifies hashes against the manifest, chunks,
embeds and builds the indexes into `./data`. The first run also downloads the embedding
model and reranker from Hugging Face into `./data/hf`.

Shortcut: embedding is the slow part on a small CPU. You can instead build locally and
copy the results: `scp -r data/processed data/index data/cache data/raw user@vps:docrag/data/`.

### 4. Start the bot

```bash
docker compose up -d bot
docker compose logs -f bot        # look for "Application started"
```

Message the bot on Telegram: `/start`, then a question.

### 5. Operate

| Task | Command |
|---|---|
| Follow logs | `docker compose logs -f bot` |
| Real usage data | `tail -f logs/usage.jsonl` (question, retrieved chunk ids, answer, latency; user ids are salted hashes) |
| Restart after config change | `docker compose up -d --build bot` |
| Update the docs snapshot | edit `corpus.commit` in `config/uniswap.yaml`, then `docker compose run --rm setup` and restart `bot` |
| Run the eval on the server | `docker compose stop bot && docker compose run --rm eval && docker compose start bot` |

The bot and the eval cannot run at the same time: Qdrant's embedded mode holds a file lock.

## Rate limits and quotas

`config/uniswap.yaml → bot_limits` caps questions per user per minute/day and a global daily
total. The global cap exists to keep the bot inside the LLM provider's free-tier daily token
quota (Groq: 200K tokens/day per model at the time of writing). When the cap is hit, users get
a clear "try again tomorrow" message instead of an error.

## Security notes

- Secrets live only in `.env` (git-ignored, `chmod 600`). The container receives them as env vars.
- The container runs as a non-root user; only `data/`, `logs/` and `eval/` are writable mounts.
- `httpx` request logging is turned down so the Telegram token never appears in logs.
- Long polling needs no inbound ports; keep the firewall closed except SSH.
