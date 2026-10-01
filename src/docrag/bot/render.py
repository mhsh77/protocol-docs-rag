"""Render a transport-neutral `Reply` as Telegram HTML (pure function, unit-tested)."""

from __future__ import annotations

from html import escape

from docrag.service import Reply

TELEGRAM_LIMIT = 4096


def to_telegram_html(reply: Reply, protocol: str = "Uniswap") -> str:
    if reply.abstained:
        parts = [f"🤷 {escape(reply.text)}"]
        if reply.closest:
            parts.append(
                f'\nClosest section I found: <a href="{escape(reply.closest.url, quote=True)}">'
                f"{escape(reply.closest.label)}</a>"
            )
        parts.append(f"\n<i>I only answer from the {escape(protocol)} docs and won't guess.</i>")
        return "\n".join(parts)

    body = escape(reply.text)
    lines = [body, "", "<b>Sources</b>"]
    for i, s in enumerate(reply.sources, 1):
        lines.append(f'[{i}] <a href="{escape(s.url, quote=True)}">{escape(s.label)}</a>')
    html = "\n".join(lines)
    if len(html) > TELEGRAM_LIMIT:  # keep the sources; trim the body
        overflow = len(html) - TELEGRAM_LIMIT + 2
        html = html.replace(
            body, escape(reply.text[: max(0, len(reply.text) - overflow - 20)]) + "…", 1
        )
    return html


START_TEXT = (
    "👋 I answer questions about the <b>{protocol}</b> developer documentation.\n\n"
    "Every answer cites the doc sections it used, with links. If the docs don't cover your "
    "question, I'll say so instead of guessing.\n\n"
    "Try: <i>How many hooks can a v4 pool have?</i>\n\n"
    "Unofficial project, not affiliated with {protocol} Labs. Not financial advice."
)
