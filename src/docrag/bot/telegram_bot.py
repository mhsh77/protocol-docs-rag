"""Telegram adapter. All logic lives in `AnswerService`; this file only moves messages.

A Discord adapter would be a sibling module with the same shape: receive text, call
`service.ask(user_id, text, channel="discord")`, render the `Reply`.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from docrag.bot.render import START_TEXT, to_telegram_html
from docrag.service import AnswerService, RateLimited

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500


def build_app(token: str, service: AnswerService, protocol: str) -> Application:  # type: ignore[type-arg]
    app = Application.builder().token(token).concurrent_updates(True).build()
    # The assistant is synchronous (CPU-bound retrieval + blocking HTTP); one at a time
    # keeps the small VPS responsive and the local Qdrant store single-threaded.
    lock = asyncio.Lock()

    async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text(
                START_TEXT.format(protocol=protocol), parse_mode=ParseMode.HTML
            )

    async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or not msg.text or update.effective_user is None:
            return
        question = msg.text.strip()
        if len(question) > MAX_QUESTION_CHARS:
            await msg.reply_text(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
            return
        await ctx.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)
        try:
            async with lock:
                reply = await asyncio.to_thread(
                    service.ask, str(update.effective_user.id), question, "telegram"
                )
        except RateLimited as e:
            when = "tomorrow" if e.scope != "user_minute" else f"in {int(e.retry_after_s) + 1}s"
            await msg.reply_text(f"⏳ You're asking faster than I can answer. Try again {when}.")
            return
        except Exception:
            log.exception("answer failed")
            await msg.reply_text("Something went wrong on my side. Please try again in a minute.")
            return
        await msg.reply_text(
            to_telegram_html(reply, protocol),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app


def run(token: str, service: AnswerService, protocol: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Request-URL logging off for both HTTP clients: Telegram URLs contain the bot token.
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    build_app(token, service, protocol).run_polling(allowed_updates=Update.ALL_TYPES)
