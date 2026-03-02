"""Slack Socket Mode handler for Nestor."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

logger = logging.getLogger(__name__)

MessageHandlerFn = Callable[[int, str, str, dict[str, str]], Awaitable[str]]


def _is_dm_channel(channel_id: str) -> bool:
    return channel_id.startswith("D")


def _conversation_int(scope: str) -> int:
    digest = hashlib.sha256(scope.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _strip_bot_mentions(text: str, bot_user_id: str) -> str:
    return re.sub(rf"<@{re.escape(bot_user_id)}>", "", text).strip()


def _thread_key(channel_id: str, thread_ts: str) -> str:
    return f"{channel_id}:{thread_ts}"


@dataclass
class SlackSocketRuntime:
    """Runtime container for a Slack Socket Mode app."""

    app: AsyncApp
    handler: AsyncSocketModeHandler

    async def start(self) -> None:
        await self.handler.connect_async()

    async def stop(self) -> None:
        if hasattr(self.handler, "close_async"):
            await self.handler.close_async()  # type: ignore[attr-defined]
            return
        if hasattr(self.handler, "disconnect_async"):
            await self.handler.disconnect_async()  # type: ignore[attr-defined]


def create_slack_socket_runtime(
    *,
    bot_token: str,
    app_token: str,
    allowed_user_ids: set[str],
    allowed_channel_ids: set[str],
    require_mention: bool,
    allow_thread_followups: bool,
    message_handler: MessageHandlerFn,
) -> SlackSocketRuntime:
    """Create a Socket Mode Slack app runtime."""
    app = AsyncApp(token=bot_token)
    active_threads: set[str] = set()
    bot_user_id_holder: dict[str, str] = {"value": ""}

    async def _bot_user_id() -> str:
        if bot_user_id_holder["value"]:
            return bot_user_id_holder["value"]
        auth = await app.client.auth_test()
        user_id = str(auth.get("user_id") or "")
        bot_user_id_holder["value"] = user_id
        return user_id

    async def _dispatch(event: dict, say) -> None:
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = str(event.get("user") or "")
        channel_id = str(event.get("channel") or "")
        if not user_id or not channel_id:
            return

        if user_id not in allowed_user_ids:
            return

        is_dm = _is_dm_channel(channel_id)
        if not is_dm and allowed_channel_ids and channel_id not in allowed_channel_ids:
            return

        text = str(event.get("text") or "").strip()
        if not text:
            return

        thread_ts = str(event.get("thread_ts") or "")
        root_ts = str(event.get("ts") or "")
        thread_or_root = thread_ts or root_ts

        bot_user_id = await _bot_user_id()
        mentioned = bool(bot_user_id and f"<@{bot_user_id}>" in text)
        thread_is_active = bool(
            thread_ts
            and allow_thread_followups
            and _thread_key(channel_id, thread_ts) in active_threads
        )

        if not is_dm and require_mention and not mentioned and not thread_is_active:
            return

        if mentioned and bot_user_id:
            text = _strip_bot_mentions(text, bot_user_id)
        text = text.strip()
        if not text:
            return

        scope = f"slack:{channel_id}:{thread_or_root or 'channel'}"
        conversation_id = _conversation_int(scope)

        context = {
            "source": "slack",
            "channel_id": channel_id,
            "thread_ts": thread_or_root or "",
        }
        reply = await message_handler(
            conversation_id,
            f"Slack-{user_id}",
            text,
            context,
        )

        if not reply.strip():
            return

        target_thread = thread_or_root or root_ts
        if target_thread:
            await say(text=reply, thread_ts=target_thread)
            active_threads.add(_thread_key(channel_id, target_thread))
        else:
            await say(text=reply)

    @app.event("message")
    async def _on_message(event, say, logger):  # type: ignore[no-untyped-def]
        try:
            await _dispatch(event, say)
        except Exception:
            logger.exception("Slack message dispatch failed")

    handler = AsyncSocketModeHandler(app, app_token)
    return SlackSocketRuntime(app=app, handler=handler)
