"""Structure-aware chunking of normalized Markdown.

Pipeline: Markdown -> top-level blocks (markdown-it tokens) -> sections (blocks under
the same heading path) -> units (blocks, or safe sub-pieces of oversized blocks) ->
chunks packed up to a token budget, with overlap only inside a section.

Invariants (tested):
* A fenced code block is never cut mid-block. Code up to `max_code_tokens` stays whole
  even if that exceeds `max_tokens`; longer code is split only at blank lines and every
  piece is re-fenced, so each piece is valid Markdown.
* A table is never cut mid-row; an oversized table is split into row groups and every
  group repeats the header, so each piece is a valid table.
* Lists split only between top-level items (nested items stay with their parent).
* Chunks never span two sections, so every chunk has exactly one heading path.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel

TokenCounter = Callable[[str], int]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\[*(])")


class ChunkingConfig(BaseModel):
    max_tokens: int = 512
    overlap_tokens: int = 64
    max_code_tokens: int = 1024
    # Adjacent sibling/parent-child sections smaller than this are merged into one chunk
    # (their sub-headings are kept inline). 0 disables merging.
    min_section_tokens: int = 0
    # A short paragraph ending in ":" is glued to the code/table/list that follows it.
    lead_in_max_tokens: int = 100


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    url: str
    source_url: str
    heading_path: list[str]
    text: str
    n_tokens: int
    ordinal: int  # position within the document

    @property
    def heading_str(self) -> str:
        return " > ".join(self.heading_path)

    def embed_text(self) -> str:
        """Text sent to the embedder/BM25: heading path gives the chunk its context."""
        return f"{self.heading_str}\n\n{self.text}"


@dataclass
class _Unit:
    text: str
    kind: str  # "prose" | "code" | "table" | "list" | "other"
    n_tokens: int
    items: list[str] = field(default_factory=list)  # top-level list items, for splitting
    cont: bool = False  # continues the previous unit's paragraph (joined with a space)
    glue_next: bool = False  # must stay in the same chunk as the following unit


@dataclass
class _Section:
    heading_path: list[str]
    blocks: list[_Unit] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return sum(b.n_tokens for b in self.blocks)


def _md() -> MarkdownIt:
    return MarkdownIt("commonmark").enable("table")


def _block_kind(tok: Token) -> str:
    if tok.type in ("fence", "code_block"):
        return "code"
    if tok.type == "table_open":
        return "table"
    if tok.type in ("bullet_list_open", "ordered_list_open"):
        return "list"
    if tok.type == "paragraph_open":
        return "prose"
    return "other"


def split_sections(text: str, title: str | None, count: TokenCounter) -> list[_Section]:
    """Group top-level blocks under their heading path."""
    lines = text.split("\n")
    tokens = _md().parse(text)
    root = [title] if title else []
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = [_Section(heading_path=list(root))]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.level != 0 or tok.map is None or tok.nesting == -1:
            i += 1
            continue
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            heading = tokens[i + 1].content.strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            # A leading H1 that repeats the frontmatter title adds nothing to the path.
            if not (level == 1 and title and heading.lower() == title.lower() and not stack):
                stack.append((level, heading))
            sections.append(_Section(heading_path=root + [h for _, h in stack]))
            i += 3
            continue
        start, end = tok.map
        block_text = "\n".join(lines[start:end]).strip("\n")
        if block_text.strip():
            kind = _block_kind(tok)
            items = _list_items(tokens, i, lines) if kind == "list" else []
            sections[-1].blocks.append(_Unit(block_text, kind, count(block_text), items))
        i = _skip_block(tokens, i)
    return [s for s in sections if s.blocks]


def _skip_block(tokens: list[Token], i: int) -> int:
    """Return the index after the block that opens at tokens[i]."""
    tok = tokens[i]
    if tok.nesting != 1:
        return i + 1
    depth = 0
    for j in range(i, len(tokens)):
        depth += tokens[j].nesting
        if depth == 0:
            return j + 1
    return len(tokens)


def _list_items(tokens: list[Token], i: int, lines: list[str]) -> list[str]:
    """Source text of each top-level item of the list opening at tokens[i]."""
    list_level = tokens[i].level
    end = _skip_block(tokens, i)
    items: list[str] = []
    for tok in tokens[i:end]:
        if tok.type == "list_item_open" and tok.level == list_level + 1 and tok.map:
            s, e = tok.map
            items.append("\n".join(lines[s:e]).rstrip())
    return items


def _split_code(unit: _Unit, cfg: ChunkingConfig, count: TokenCounter) -> list[_Unit]:
    if unit.n_tokens <= cfg.max_code_tokens:
        return [unit]
    body = unit.text.split("\n")
    opener, closer = body[0], body[-1] if len(body) > 1 else body[0][:3]
    inner = body[1:-1]
    # Paragraph-like groups of code separated by blank lines.
    groups: list[list[str]] = [[]]
    for line in inner:
        groups[-1].append(line)
        if not line.strip():
            groups.append([])
    pieces: list[list[str]] = [[]]
    for g in groups:
        candidate = pieces[-1] + g
        if pieces[-1] and count("\n".join(candidate)) > cfg.max_code_tokens:
            pieces.append(list(g))
        else:
            pieces[-1] = candidate
    n = len(pieces)
    if n == 1:  # no blank line to split at: keep the block whole rather than cut it
        return [unit]
    out = []
    for k, p in enumerate(pieces, 1):
        label = f"// (code block part {k}/{n})"
        text = "\n".join([opener, label, *p]).rstrip("\n") + "\n" + closer
        out.append(_Unit(text, "code", count(text)))
    return out


def _split_table(unit: _Unit, cfg: ChunkingConfig, count: TokenCounter) -> list[_Unit]:
    if unit.n_tokens <= cfg.max_tokens:
        return [unit]
    rows = unit.text.split("\n")
    header, rows = rows[:2], rows[2:]
    out: list[_Unit] = []
    cur: list[str] = []
    for row in rows:
        if cur and count("\n".join(header + cur + [row])) > cfg.max_tokens:
            text = "\n".join(header + cur)
            out.append(_Unit(text, "table", count(text)))
            cur = []
        cur.append(row)
    if cur:
        text = "\n".join(header + cur)
        out.append(_Unit(text, "table", count(text)))
    return out


def _split_prose(unit: _Unit, cfg: ChunkingConfig, count: TokenCounter) -> list[_Unit]:
    if unit.n_tokens <= cfg.max_tokens:
        return [unit]
    sents = [s for s in _SENTENCE_RE.split(unit.text) if s.strip()]
    return [_Unit(s, unit.kind, count(s), cont=k > 0) for k, s in enumerate(sents)]


def to_units(block: _Unit, cfg: ChunkingConfig, count: TokenCounter) -> list[_Unit]:
    """Break an oversized block into structure-preserving units."""
    if block.kind == "code":
        return _split_code(block, cfg, count)
    if block.kind == "table":
        return _split_table(block, cfg, count)
    if block.kind == "list" and block.n_tokens > cfg.max_tokens:
        units: list[_Unit] = []
        for it in block.items:
            u = _Unit(it, "list", count(it))
            units.extend(_split_prose(u, cfg, count) if u.n_tokens > cfg.max_tokens else [u])
        return units
    if block.kind == "prose":
        return _split_prose(block, cfg, count)
    return [block]


def _overlap_tail(units: list[_Unit], budget: int, count: TokenCounter) -> list[_Unit]:
    """Trailing context from the previous chunk, at most `budget` tokens, prose/list only."""
    if budget <= 0:
        return []
    tail: list[_Unit] = []
    used = 0
    for u in reversed(units):
        if u.kind not in ("prose", "list"):
            break
        if used + u.n_tokens <= budget:
            tail.insert(0, u)
            used += u.n_tokens
            continue
        # Take whole trailing sentences of this unit that still fit.
        sents = [s for s in _SENTENCE_RE.split(u.text) if s.strip()]
        picked: list[str] = []
        for s in reversed(sents):
            n = count(s)
            if used + n > budget:
                break
            picked.insert(0, s)
            used += n
        if picked:
            text = " ".join(picked)
            tail.insert(0, _Unit(text, u.kind, count(text), cont=u.cont))
        break
    return tail


def _glue(units: list[_Unit], cfg: ChunkingConfig, count: TokenCounter) -> list[_Unit]:
    """Merge lead-ins ("...looks like this:") and inline sub-headings into the next unit."""
    out: list[_Unit] = []
    pending: _Unit | None = None
    for u in units:
        is_lead_in = (
            u.kind == "prose"
            and not u.cont
            and u.text.rstrip().endswith(":")
            and u.n_tokens <= cfg.lead_in_max_tokens
        )
        if pending is not None:
            text = f"{pending.text}\n\n{u.text}"
            u = _Unit(text, u.kind, count(text), u.items, glue_next=u.glue_next)
            pending = None
        if u.glue_next or is_lead_in:
            pending = u
            continue
        out.append(u)
    if pending is not None:
        out.append(pending)
    return out


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def merge_small_sections(
    sections: list[_Section], cfg: ChunkingConfig, count: TokenCounter
) -> list[_Section]:
    """Merge runs of small adjacent sections that are siblings or parent/child.

    The merged section's heading path is the members' common prefix; each member's
    remaining heading is kept inline as a bold line so no context is lost.
    """
    if cfg.min_section_tokens <= 0:
        return sections
    groups: list[list[_Section]] = []
    for sec in sections:
        if groups:
            group = groups[-1]
            prefix = group[0].heading_path
            for g in group[1:]:
                prefix = _common_prefix(prefix, g.heading_path)
            new_prefix = _common_prefix(prefix, sec.heading_path)
            related = len(new_prefix) >= max(1, min(len(prefix), len(sec.heading_path)) - 1)
            group_tokens = sum(g.n_tokens for g in group)
            # A small section may join a related group; a large one may only join a small
            # group without widening its heading path (i.e. as a child), so big unrelated
            # siblings never dilute a group's context.
            allowed = sec.n_tokens < cfg.min_section_tokens or (
                group_tokens < cfg.min_section_tokens and new_prefix == prefix
            )
            fits = group_tokens + sec.n_tokens <= cfg.max_tokens
            if related and allowed and fits:
                group.append(sec)
                continue
        groups.append([sec])

    merged: list[_Section] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        prefix = group[0].heading_path
        for g in group[1:]:
            prefix = _common_prefix(prefix, g.heading_path)
        blocks: list[_Unit] = []
        for g in group:
            rest = g.heading_path[len(prefix) :]
            if rest:
                text = f"**{' > '.join(rest)}**"
                blocks.append(_Unit(text, "heading", count(text), glue_next=True))
            blocks.extend(g.blocks)
        merged.append(_Section(heading_path=list(prefix), blocks=blocks))
    return merged


def pack_section(
    section: _Section, cfg: ChunkingConfig, count: TokenCounter
) -> list[tuple[str, int]]:
    """Pack a section's units into chunk texts. Returns (text, n_tokens) pairs."""
    units = _glue([u for b in section.blocks for u in to_units(b, cfg, count)], cfg, count)
    chunks: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_new = 0  # tokens of units not carried over as overlap
    for u in units:
        cur_tokens = sum(x.n_tokens for x in cur)
        if cur and cur_new > 0 and cur_tokens + u.n_tokens > cfg.max_tokens:
            chunks.append(cur)
            cur = _overlap_tail(cur, cfg.overlap_tokens, count)
            if sum(x.n_tokens for x in cur) + u.n_tokens > cfg.max_tokens:
                cur = []
            cur_new = 0
        cur.append(u)
        cur_new += u.n_tokens
    if cur and cur_new > 0:
        chunks.append(cur)
    out = []
    for c in chunks:
        text = c[0].text + "".join((" " if u.cont else "\n\n") + u.text for u in c[1:])
        out.append((text, count(text)))
    return out


def chunk_document(
    *,
    text: str,
    title: str | None,
    doc_id: str,
    doc_slug: str,
    url: str,
    source_url: str,
    cfg: ChunkingConfig,
    count: TokenCounter,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = merge_small_sections(split_sections(text, title, count), cfg, count)
    for section in sections:
        for chunk_text, n in pack_section(section, cfg, count):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_slug}#{ordinal:03d}",
                    doc_id=doc_id,
                    url=url,
                    source_url=source_url,
                    heading_path=section.heading_path or [doc_slug],
                    text=chunk_text,
                    n_tokens=n,
                    ordinal=ordinal,
                )
            )
    return chunks
