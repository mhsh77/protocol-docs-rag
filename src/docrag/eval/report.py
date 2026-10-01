"""Aggregate saved run records into the results table (`results.json` + `results.md`).

Everything here is recomputed from `eval/runs/<run>/records/*.jsonl`, so every number in
the README traces back to raw, inspectable records.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from docrag.eval.metrics import (
    bootstrap_ci,
    evidence_recall_at_k,
    hit_at_k,
    mean,
    paired_bootstrap_diff,
    percentile,
    reciprocal_rank,
)
from docrag.eval.runner import Record

# Questions where the right behaviour is to decline.
ABSTAIN_EXPECTED = "abstain"


def load_records(run_dir: Path) -> dict[str, list[Record]]:
    out: dict[str, list[Record]] = {}
    for path in sorted((run_dir / "records").glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            recs = [Record.model_validate_json(line) for line in f if line.strip()]
        out[path.stem] = sorted(recs, key=lambda r: r.qid)
    return out


def _gold(r: Record) -> set[str]:
    return {cid for g in r.gold_groups for cid in g}


def per_question(r: Record) -> dict[str, float | None]:
    """Per-question metric values (None = not applicable to this question)."""
    answerable = r.expected != ABSTAIN_EXPECTED
    has_gold = answerable and bool(_gold(r))
    v = r.verdict
    claims = v.claims if v else []
    n_claims = len(claims)
    n_unsup = sum(not c.supported for c in claims)
    return {
        "hit@1": hit_at_k(r.retrieved_ids, _gold(r), 1) if has_gold else None,
        "hit@5": hit_at_k(r.retrieved_ids, _gold(r), 5) if has_gold else None,
        "mrr@10": reciprocal_rank(r.retrieved_ids, _gold(r)) if has_gold else None,
        "evidence_recall@5": (
            evidence_recall_at_k(r.retrieved_ids, [set(g) for g in r.gold_groups], 5)
            if has_gold
            else None
        ),
        # Correctness on answerable questions; declining counts as 0.
        "correctness": (
            (v.correctness / 2 if v else 0.0) if answerable and (v or r.abstained) else None
        ),
        "fully_correct": (
            (1.0 if v and v.correctness == 2 else 0.0)
            if answerable and (v or r.abstained)
            else None
        ),
        # Claim-level hallucination, over answered questions that the judge graded.
        "unsupported_claims": float(n_unsup) if v and not r.abstained else None,
        "total_claims": float(n_claims) if v and not r.abstained else None,
        "any_unsupported": (1.0 if n_unsup else 0.0)
        if v and not r.abstained and n_claims
        else None,
        "correct_abstention": (1.0 if r.abstained else 0.0) if not answerable else None,
        "false_abstention": (1.0 if r.abstained else 0.0) if answerable else None,
        "citation_valid": (
            (1.0 if r.citation_valid else 0.0)
            if not r.abstained and r.citation_valid is not None and r.citations_requested
            else None
        ),
        "citation_first_try_valid": (
            (1.0 if r.citation_first_attempt_valid else 0.0)
            if r.citation_first_attempt_valid is not None
            else None
        ),
        # Answers in which every claim sentence carries a citation (measured, not enforced).
        "fully_cited": (
            (1.0 if r.n_uncited_sentences == 0 else 0.0)
            if not r.abstained and r.n_uncited_sentences is not None and r.citations_requested
            else None
        ),
        "latency_s": r.retrieval_s + r.generation_s,
        "tokens": float(r.input_tokens + r.output_tokens),
        "judge_tokens": float(r.judge_tokens),
    }


def _vals(rows: list[dict[str, float | None]], key: str) -> list[float]:
    return [x for row in rows if (x := row[key]) is not None]


def summarize(records: list[Record]) -> dict[str, object]:
    rows = [per_question(r) for r in records]
    out: dict[str, object] = {"n_questions": len(records)}
    for key in (
        "hit@1",
        "hit@5",
        "mrr@10",
        "evidence_recall@5",
        "correctness",
        "fully_correct",
        "any_unsupported",
        "correct_abstention",
        "false_abstention",
        "citation_valid",
        "citation_first_try_valid",
        "fully_cited",
    ):
        vals = _vals(rows, key)
        lo, hi = bootstrap_ci(vals)
        out[key] = {"mean": mean(vals), "ci95": [lo, hi], "n": len(vals)}
    unsup, total = _vals(rows, "unsupported_claims"), _vals(rows, "total_claims")
    out["hallucination_rate"] = {
        "mean": sum(unsup) / sum(total) if sum(total) else float("nan"),
        "unsupported_claims": int(sum(unsup)),
        "total_claims": int(sum(total)),
        "n": len(total),
    }
    lat = _vals(rows, "latency_s")
    out["latency_s"] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95)}
    out["tokens_per_question"] = mean(_vals(rows, "tokens"))
    out["judge_tokens_per_question"] = mean(_vals(rows, "judge_tokens"))
    out["by_category"] = {
        cat: {
            "n": len(rs),
            "correctness": mean(_vals([per_question(r) for r in rs], "correctness")),
            "correct_abstention": mean(_vals([per_question(r) for r in rs], "correct_abstention")),
        }
        for cat in sorted({r.category for r in records})
        if (rs := [r for r in records if r.category == cat])
    }
    return out


def compare(base: list[Record], other: list[Record], key: str) -> tuple[float, float, float] | None:
    """Paired difference (other - base) over questions where the metric applies to both."""
    b = {r.qid: per_question(r)[key] for r in base}
    o = {r.qid: per_question(r)[key] for r in other}
    common = [q for q in b if q in o and b[q] is not None and o[q] is not None]
    if not common:
        return None
    return paired_bootstrap_diff([b[q] for q in common], [o[q] for q in common])  # type: ignore[misc]


COLUMNS: Sequence[tuple[str, str, bool]] = (
    # (summary key, column title, lower_is_better)
    ("hit@5", "Hit@5", False),
    ("mrr@10", "MRR@10", False),
    ("correctness", "Answer correctness", False),
    ("hallucination_rate", "Hallucination rate (claims)", True),
    ("correct_abstention", "Correct abstention (unanswerable)", False),
    ("false_abstention", "False abstention (answerable)", True),
    ("citation_valid", "Citation validity", False),
)


def _fmt(summary: dict[str, object], key: str) -> str:
    v = summary[key]
    assert isinstance(v, dict)
    m = v["mean"]
    if m != m:  # NaN
        return "n/a"
    if key == "hallucination_rate":
        return f"{m:.1%} ({v['unsupported_claims']}/{v['total_claims']})"
    lo, hi = v["ci95"]
    return f"{m:.1%} [{lo:.0%}, {hi:.0%}]"


def render_markdown(summaries: dict[str, dict[str, object]], run_dir: Path, split: str) -> str:
    lines = [
        f"Run `{run_dir.name}` · split `{split}` · 95% bootstrap CIs in brackets.",
        "",
        "| Configuration | "
        + " | ".join(c[1] for c in COLUMNS)
        + " | Latency p50 / p95 | Tokens/q |",
        "|---|" + "---|" * len(COLUMNS) + "---|---|",
    ]
    for name, s in summaries.items():
        lat = s["latency_s"]
        assert isinstance(lat, dict)
        cells = [_fmt(s, key) for key, _, _ in COLUMNS]
        lines.append(
            f"| `{name}` | "
            + " | ".join(cells)
            + f" | {lat['p50']:.1f}s / {lat['p95']:.1f}s | {s['tokens_per_question']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, split: str = "test") -> dict[str, dict[str, object]]:
    recs = load_records(run_dir)
    recs = {k: [r for r in v if split == "all" or r.split == split] for k, v in recs.items()}
    summaries = {name: summarize(rs) for name, rs in recs.items() if rs}
    # Paired comparisons of every configuration against each baseline that is present.
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    keys = (
        "hit@5",
        "mrr@10",
        "correctness",
        "any_unsupported",
        "correct_abstention",
        "false_abstention",
    )
    for base in ("naive_rag", "baseline_dense"):
        if base not in recs:
            continue
        comparisons[base] = {}
        for name, rs in recs.items():
            if name == base:
                continue
            comparisons[base][name] = {}
            for key in keys:
                diff = compare(recs[base], rs, key)
                if diff:
                    comparisons[base][name][key] = {"diff": diff[0], "ci95": [diff[1], diff[2]]}
    (run_dir / "results.json").write_text(
        json.dumps({"split": split, "summaries": summaries, "vs_baseline": comparisons}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "results.md").write_text(
        render_markdown(summaries, run_dir, split), encoding="utf-8"
    )
    return summaries


README_START = "<!-- results:start (generated by `rag report --update-readme`; do not edit) -->"
README_END = "<!-- results:end -->"


def update_readme_block(readme: Path, table_md: str) -> None:
    """Replace the generated results block in the README (between the two markers)."""
    text = readme.read_text(encoding="utf-8")
    i, j = text.index(README_START), text.index(README_END)
    readme.write_text(
        text[: i + len(README_START)] + "\n" + table_md.strip() + "\n" + text[j:],
        encoding="utf-8",
        newline="\n",
    )
