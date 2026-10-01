"""Citation parsing and validation.

An answer is citation-valid when:
* every inline marker [S<n>] refers to a source that was actually retrieved, and
* a non-abstaining answer cites at least one source, and
* every sentence that makes a claim carries a citation (sentence coverage).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_MARKER_RE = re.compile(r"\[S(\d+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"(])")
_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


class CitationCheck(BaseModel):
    valid: bool
    cited: list[str]  # aliases in first-appearance order, e.g. ["S1", "S3"]
    unknown: list[str]  # cited aliases that were not in the retrieved set
    uncited_sentences: list[str]
    reason: str | None = None


def cited_aliases(text: str) -> list[str]:
    return list(dict.fromkeys(f"S{m}" for m in _MARKER_RE.findall(text)))


def uncited_sentences(text: str, min_words: int = 4) -> list[str]:
    """Sentences of at least `min_words` words without any [S<n>] marker.

    Code blocks are ignored, and a sentence ending in ':' that introduces a following
    cited list or code block is not counted as a claim on its own.
    """
    prose = _CODE_RE.sub(" ", text)
    out = []
    for raw in re.split(r"\n\s*\n", prose):
        for sent in _SENT_SPLIT_RE.split(raw.strip()):
            s = sent.strip()
            if not s or s.endswith(":"):
                continue
            if len(s.split()) >= min_words and not _MARKER_RE.search(s):
                out.append(s)
    return out


def check_citations(
    answer: str, n_sources: int, abstained: bool, require_sentence_coverage: bool = True
) -> CitationCheck:
    cited = cited_aliases(answer)
    allowed = {f"S{i + 1}" for i in range(n_sources)}
    unknown = [a for a in cited if a not in allowed]
    uncited = [] if abstained else uncited_sentences(answer)
    reason = None
    if unknown:
        reason = f"cites sources that were not retrieved: {', '.join(unknown)}"
    elif not abstained and not cited:
        reason = "answer has no citations"
    elif require_sentence_coverage and uncited:
        reason = f"{len(uncited)} sentence(s) without a citation"
    return CitationCheck(
        valid=reason is None,
        cited=cited,
        unknown=unknown,
        uncited_sentences=uncited,
        reason=reason,
    )


def strip_unknown_markers(answer: str, n_sources: int) -> str:
    def keep(m: re.Match[str]) -> str:
        return m.group(0) if 1 <= int(m.group(1)) <= n_sources else ""

    return _MARKER_RE.sub(keep, answer)
