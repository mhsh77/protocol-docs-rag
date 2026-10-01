"""Download a pinned snapshot of a docs repo and write a provenance manifest.

Each fetched file gets a manifest row with its source URL (pinned to the commit),
the published page URL, retrieval timestamp, and sha256, so every chunk and every
eval label can be traced back to an exact upstream file.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from docrag.config import CorpusConfig

MANIFEST_NAME = "manifest.jsonl"
USER_AGENT = "docrag-corpus-fetcher (portfolio project)"


class ManifestEntry(BaseModel):
    doc_id: str  # repo-relative path, stable across runs
    local_path: str  # path relative to the corpus raw dir
    source_url: str
    published_url: str
    repo: str
    commit: str
    license: str
    retrieved_at: str
    sha256: str
    bytes: int


def list_repo_files(client: httpx.Client, cfg: CorpusConfig) -> list[str]:
    url = f"https://api.github.com/repos/{cfg.repo}/git/trees/{cfg.commit}?recursive=1"
    resp = client.get(url)
    resp.raise_for_status()
    tree = resp.json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing was truncated; narrow include_prefixes.")
    return sorted(e["path"] for e in tree["tree"] if e["type"] == "blob" and cfg.wants(e["path"]))


def _download(
    client: httpx.Client, cfg: CorpusConfig, repo_path: str, out_dir: Path
) -> ManifestEntry:
    resp = client.get(cfg.raw_url(repo_path))
    resp.raise_for_status()
    content = resp.content
    local = Path(repo_path.removeprefix(cfg.strip_prefix))
    dest = out_dir / local
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return ManifestEntry(
        doc_id=repo_path,
        local_path=local.as_posix(),
        source_url=cfg.source_url(repo_path),
        published_url=cfg.published_url(repo_path),
        repo=cfg.repo,
        commit=cfg.commit,
        license=cfg.license,
        retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )


def fetch_corpus(cfg: CorpusConfig, out_dir: Path, max_workers: int = 4) -> list[ManifestEntry]:
    """Fetch all corpus files at the pinned commit into `out_dir` and write the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True
    ) as client:
        paths = list_repo_files(client, cfg)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            entries = list(pool.map(lambda p: _download(client, cfg, p, out_dir), paths))
        if cfg.license_file:
            resp = client.get(cfg.raw_url(cfg.license_file))
            resp.raise_for_status()
            (out_dir / "UPSTREAM_LICENSE").write_bytes(resp.content)
    write_manifest(entries, out_dir / MANIFEST_NAME)
    return entries


def write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in sorted(entries, key=lambda e: e.doc_id):
            f.write(e.model_dump_json() + "\n")


def read_manifest(path: Path) -> list[ManifestEntry]:
    with open(path, encoding="utf-8") as f:
        return [ManifestEntry.model_validate(json.loads(line)) for line in f if line.strip()]


def verify_corpus(raw_dir: Path) -> list[str]:
    """Return a list of problems (missing files, hash mismatches, unlisted files). Empty = OK."""
    entries = read_manifest(raw_dir / MANIFEST_NAME)
    problems: list[str] = []
    listed = set()
    for e in entries:
        p = raw_dir / e.local_path
        listed.add(p.resolve())
        if not p.exists():
            problems.append(f"missing: {e.local_path}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != e.sha256:
            problems.append(f"hash mismatch: {e.local_path}")
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".mdx"} and p.resolve() not in listed:
            problems.append(f"not in manifest: {p.relative_to(raw_dir).as_posix()}")
    return problems
