"""Normalize MDX into plain CommonMark + GFM tables before chunking.

MDX mixes Markdown with JSX components. Components carry little retrievable text but
break Markdown parsing, so we rewrite the ones that matter and drop the rest:

* frontmatter        -> returned as metadata (title, description)
* <Callout type=..>  -> a bold "Note:" / "Warning:" lead-in, body kept
* <Card title href description/> -> a bullet "**title**: description (href)"
* <ImplementationTab value="x"> -> headings inside the tab get a " (x)" suffix so that
  repeated "Step 1" headings across tabs stay distinguishable in heading paths
* <summary>X</summary> -> "**X**"
* other component / layout tags (div, details, Tabs, ...) -> removed, inner text kept
* {/* MDX comments */} -> removed

Nothing inside fenced code blocks is touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
_HTML_TAGS = "div|details|span|section|p|br|img|a|figure|figcaption|iframe|video|source|center"
# A line that is only a tag (open, close or self-closing), for components (Capitalized)
# or known layout HTML tags. Multi-line attribute tags are joined before matching.
_TAG_ONLY_RE = re.compile(rf"^\s*</?(?:[A-Z][\w.]*|(?:{_HTML_TAGS})\b)[^<>]*/?>\s*$", re.DOTALL)
_TAG_START_RE = re.compile(rf"^\s*<(?:[A-Z][\w.]*|(?:{_HTML_TAGS})\b)")
_ATTR_RE = re.compile(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|\{"([^"]*)"\})')
_SUMMARY_RE = re.compile(r"^\s*<summary>(.*?)</summary>\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

_CALLOUT_LABELS = {"warn": "Warning", "warning": "Warning", "danger": "Warning", "error": "Warning"}


@dataclass
class NormalizedDoc:
    text: str
    title: str | None = None
    description: str | None = None
    meta: dict[str, object] = field(default_factory=dict)


def _attrs(tag: str) -> dict[str, str]:
    return {m[0]: m[1] or m[2] or m[3] for m in _ATTR_RE.findall(tag)}


def _tag_name(tag: str) -> str:
    m = re.match(r"^\s*</?\s*([\w.]+)", tag)
    return m.group(1) if m else ""


def split_frontmatter(source: str) -> tuple[dict[str, object], str]:
    m = _FRONTMATTER_RE.match(source)
    if not m:
        return {}, source
    data = yaml.safe_load(m.group(1)) or {}
    return (data if isinstance(data, dict) else {}), source[m.end() :]


def normalize_mdx(source: str) -> NormalizedDoc:
    meta, body = split_frontmatter(source.replace("\r\n", "\n"))
    lines = body.split("\n")
    out: list[str] = []
    fence: str | None = None
    tab_stack: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        fm = _FENCE_RE.match(line)
        if fence is not None:
            out.append(line)
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= len(fence):
                fence = None
            i += 1
            continue
        if fm:
            fence = fm.group(1)
            out.append(line)
            i += 1
            continue

        # Gather a multi-line JSX tag (e.g. <Card\n title=...\n />) into one string.
        if _TAG_START_RE.match(line) and ">" not in line:
            j = i
            buf = [line]
            while ">" not in buf[-1] and j + 1 < len(lines):
                j += 1
                buf.append(lines[j])
            joined = " ".join(s.strip() for s in buf)
            if _TAG_ONLY_RE.match(joined):
                out.extend(_rewrite_tag(joined, tab_stack))
                i = j + 1
                continue

        # MDX comments, possibly spanning lines (outside code only).
        if "{/*" in line:
            j = i
            while "*/}" not in line and j + 1 < len(lines):
                j += 1
                line = line + "\n" + lines[j]
            line = _MDX_COMMENT_RE.sub("", line)
            i = j
            if not line.strip():
                i += 1
                continue

        summary = _SUMMARY_RE.match(line)
        if summary:
            out.append(f"**{summary.group(1).strip()}**")
        elif _TAG_ONLY_RE.match(line):
            out.extend(_rewrite_tag(line, tab_stack))
        else:
            h = _HEADING_RE.match(line)
            if h and tab_stack:
                out.append(f"{h.group(1)} {h.group(2)} ({tab_stack[-1]})")
            else:
                out.append(line)
        i += 1

    if fence is not None:  # unterminated fence in the source: close it at end of file
        out.append(fence)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    title = meta.get("title")
    desc = meta.get("description")
    return NormalizedDoc(
        text=text,
        title=str(title) if title else None,
        description=str(desc) if desc else None,
        meta=meta,
    )


def _rewrite_tag(tag: str, tab_stack: list[str]) -> list[str]:
    """Return replacement lines for a tag-only line. Updates the tab context stack."""
    name = _tag_name(tag)
    closing = tag.lstrip().startswith("</")
    attrs = _attrs(tag)
    if name in ("ImplementationTab", "TabsContent", "Tab"):
        if closing:
            if tab_stack:
                tab_stack.pop()
        elif not tag.rstrip().endswith("/>"):
            tab_stack.append(
                attrs.get("value") or attrs.get("title") or attrs.get("label") or "tab"
            )
        return [""]
    if name == "Callout" and not closing:
        label = _CALLOUT_LABELS.get(attrs.get("type", "").lower(), "Note")
        return ["", f"**{label}:**", ""]
    if name == "Card" and "title" in attrs:
        desc = attrs.get("description", "")
        href = attrs.get("href")
        item = f"- **{attrs['title']}**" + (f": {desc}" if desc else "")
        return [item + (f" ({href})" if href else "")]
    return [""]
