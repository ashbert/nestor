"""Telegram bot handler for Nestor — the butler from Tintin.

Whitelist-enforced bot that routes messages to an LLM handler and
supports /start, /today, and /week commands.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

# Type aliases for the injected handlers.
MessageHandlerFn: TypeAlias = Callable[[int, str, str], Awaitable[str]]
ScheduleHandlerFn: TypeAlias = Callable[[int], Awaitable[str]]

# Telegram enforces a 4 096-character limit per message.
_MAX_MESSAGE_LENGTH = 4096

_START_TEXT = (
    "Good day. I am Nestor, at your service.\n\n"
    "You may address me freely — I shall do my utmost to assist, "
    "though I make no promises regarding matters involving llamas.\n\n"
    "Available commands:\n"
    "/today — today's agenda\n"
    "/week  — the week ahead"
)

_ERROR_TEXT = (
    "I do beg your pardon — an unforeseen difficulty has arisen "
    "on my end. Rest assured I shall look into it promptly."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _send_long_message(update: Update, text: str) -> None:
    """Send *text*, splitting into chunks if it exceeds Telegram's limit."""
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id

    for offset in range(0, len(text), _MAX_MESSAGE_LENGTH):
        chunk = text[offset : offset + _MAX_MESSAGE_LENGTH]
        await update.get_bot().send_message(chat_id=chat_id, text=chunk)


def _is_allowed(user_id: int | None, allowed_ids: set[int]) -> bool:
    """Return True when *user_id* is in the whitelist."""
    return user_id is not None and user_id in allowed_ids


def _is_private_chat(update: Update) -> bool:
    """Return True only for Telegram private chats."""
    chat = update.effective_chat
    return bool(chat and chat.type == ChatType.PRIVATE)


# ---------------------------------------------------------------------------
# Handler factories (closures over configuration)
# ---------------------------------------------------------------------------

def _make_start_handler(allowed_ids: set[int]):
    """Return a /start command handler."""

    async def _start(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not _is_private_chat(update):
            return
        if not update.effective_user or not _is_allowed(
            update.effective_user.id, allowed_ids
        ):
            return  # silently ignore

        if update.message:
            await update.message.reply_text(_START_TEXT)

    return _start


def _make_today_handler(
    allowed_ids: set[int],
    today_handler: ScheduleHandlerFn,
):
    """Return a /today command handler."""

    async def _today(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not _is_private_chat(update):
            return
        user = update.effective_user
        if not user or not _is_allowed(user.id, allowed_ids):
            return

        try:
            if update.effective_chat:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action=ChatAction.TYPING,
                )
            response = await today_handler(user.id)
            await _send_long_message(update, response)
        except Exception:
            logger.exception("/today handler failed for user %s", user.id)
            if update.message:
                await update.message.reply_text(_ERROR_TEXT)

    return _today


def _make_week_handler(
    allowed_ids: set[int],
    week_handler: ScheduleHandlerFn,
):
    """Return a /week command handler."""

    async def _week(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not _is_private_chat(update):
            return
        user = update.effective_user
        if not user or not _is_allowed(user.id, allowed_ids):
            return

        try:
            if update.effective_chat:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action=ChatAction.TYPING,
                )
            response = await week_handler(user.id)
            await _send_long_message(update, response)
        except Exception:
            logger.exception("/week handler failed for user %s", user.id)
            if update.message:
                await update.message.reply_text(_ERROR_TEXT)

    return _week


def _make_message_handler(
    allowed_ids: set[int],
    message_handler: MessageHandlerFn,
):
    """Return a plain-text message handler."""

    async def _handle_message(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not _is_private_chat(update):
            return
        user = update.effective_user
        if not user or not _is_allowed(user.id, allowed_ids):
            return

        text = (update.message.text if update.message else None) or ""
        if not text.strip():
            return

        try:
            if update.effective_chat:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action=ChatAction.TYPING,
                )
            response = await message_handler(
                user.id,
                user.first_name or "",
                text,
            )
            await _send_long_message(update, response)
        except Exception:
            logger.exception(
                "Message handler failed for user %s", user.id
            )
            if update.message:
                await update.message.reply_text(_ERROR_TEXT)

    return _handle_message


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_bot(
    token: str,
    allowed_ids: set[int],
    message_handler: MessageHandlerFn,
    today_handler: ScheduleHandlerFn,
    week_handler: ScheduleHandlerFn,
) -> Application:
    """Build and return a fully-configured :class:`Application`.

    Parameters
    ----------
    token:
        Telegram Bot API token.
    allowed_ids:
        Set of Telegram user IDs permitted to interact with the bot.
    message_handler:
        ``async (user_id, first_name, text) -> response`` callable for
        free-form messages.
    today_handler:
        ``async (user_id) -> response`` callable for the /today command.
    week_handler:
        ``async (user_id) -> response`` callable for the /week command.

    Returns
    -------
    Application
        Ready to be started via ``application.run_polling()`` or similar.
    """
    application = Application.builder().token(token).build()

    application.add_handler(
        CommandHandler("start", _make_start_handler(allowed_ids))
    )
    application.add_handler(
        CommandHandler("today", _make_today_handler(allowed_ids, today_handler))
    )
    application.add_handler(
        CommandHandler("week", _make_week_handler(allowed_ids, week_handler))
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            _make_message_handler(allowed_ids, message_handler),
        )
    )

    logger.info(
        "Nestor bot configured — %d user(s) on the whitelist.",
        len(allowed_ids),
    )
    return application
