"""Nestor's brain — orchestrates LLM, tools, and memory.

The :class:`NestorBrain` runs the agentic loop: it loads conversation
history, calls the LLM, executes any requested tools, feeds results
back, and persists the exchange.
"""

from __future__ import annotations

import secrets
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from nestor.llm import LLMProvider, LLMResponse, ToolCall
from nestor.memory import MemoryStore
from nestor.tools import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 5
_CONFIRMATION_TTL = timedelta(minutes=15)
_SENSITIVE_TOOLS = {
    "send_email",
    "create_calendar_event",
    "delete_calendar_event",
    "create_note",
    "append_note",
}


@dataclass(slots=True)
class PendingAction:
    token: str
    tool_calls: list[ToolCall]
    created_at: datetime


class NestorBrain:
    """High-level orchestrator that ties the LLM to tools and memory."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        memory: MemoryStore,
        system_prompt: str,
    ) -> None:
        self._llm = llm
        self._tools = tool_registry
        self._memory = memory
        self._system_prompt = system_prompt
        self._pending_actions: dict[int, PendingAction] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_datetime(self, prompt: str) -> str:
        """Replace a ``{current_datetime}`` placeholder with local time."""
        tz_name = os.environ.get("NESTOR_TIMEZONE", "America/Los_Angeles")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("America/Los_Angeles")
        now = datetime.now(tz=tz)
        dt_str = now.strftime(f"%A, %d %B %Y, %H:%M {tz_name}")
        return prompt.replace("{current_datetime}", dt_str)

    def _build_messages(
        self,
        user_id: int,
        user_name: str,
        message: str,
    ) -> list[dict[str, Any]]:
        """Load history and assemble the full message list."""
        history_rows = self._memory.get_recent_messages(user_id, limit=50)
        messages: list[dict[str, Any]] = []
        for row in history_rows:
            messages.append({"role": row["role"], "content": row["content"]})
        messages.append({"role": "user", "content": f"[{user_name}]: {message}"})
        return messages

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[dict[str, Any]]:
        """Run every requested tool and return result messages."""
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            logger.info("Executing tool %s (id=%s)", tc.name, tc.id)
            try:
                output = await self._tools.execute(tc.name, tc.arguments)
                content = json.dumps(output) if not isinstance(output, str) else output
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool %s failed", tc.name)
                content = json.dumps({"error": str(exc)})
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )
        return results

    def _persist_exchange(
        self, user_id: int, user_name: str, user_message: str, assistant_message: str
    ) -> None:
        self._memory.save_message(user_id, "user", f"[{user_name}]: {user_message}")
        self._memory.save_message(user_id, "assistant", assistant_message)

    def _get_pending_action(self, user_id: int) -> PendingAction | None:
        pending = self._pending_actions.get(user_id)
        if not pending:
            return None
        if datetime.now(timezone.utc) - pending.created_at > _CONFIRMATION_TTL:
            self._pending_actions.pop(user_id, None)
            return None
        return pending

    @staticmethod
    def _extract_confirmation_token(message: str) -> str | None:
        text = message.strip()
        lowered = text.lower()
        for prefix in ("confirm ", "approve "):
            if lowered.startswith(prefix):
                token = text[len(prefix) :].strip()
                return token or None
        return None

    @staticmethod
    def _is_cancel_message(message: str) -> bool:
        return message.strip().lower() in {"cancel", "deny", "reject", "abort"}

    def _stage_pending_action(self, user_id: int, tool_calls: list[ToolCall]) -> str:
        token = f"{secrets.randbelow(1_000_000):06d}"
        self._pending_actions[user_id] = PendingAction(
            token=token,
            tool_calls=tool_calls,
            created_at=datetime.now(timezone.utc),
        )

        lines = [
            "I need explicit confirmation before executing these side-effect actions:",
        ]
        for tc in tool_calls:
            args = json.dumps(tc.arguments, ensure_ascii=True)
            if len(args) > 220:
                args = args[:220] + "... [truncated]"
            lines.append(f"- `{tc.name}` with args: {args}")
        lines.append("")
        lines.append(f"Reply with `confirm {token}` to proceed.")
        lines.append("Reply with `cancel` to discard this request.")
        lines.append("This confirmation expires in 15 minutes.")
        return "\n".join(lines)

    @staticmethod
    def _format_confirmed_results(tool_calls: list[ToolCall], tool_results: list[dict[str, Any]]) -> str:
        lines = ["Confirmed. I executed the requested actions:"]
        for tc, result in zip(tool_calls, tool_results):
            content = str(result.get("content", "")).strip()
            if len(content) > 450:
                content = content[:450] + "... [truncated]"
            lines.append(f"- `{tc.name}`: {content}")
        return "\n".join(lines)

    async def _maybe_handle_confirmation(
        self, user_id: int, user_name: str, message: str
    ) -> str | None:
        pending = self._get_pending_action(user_id)
        token = self._extract_confirmation_token(message)

        if not pending:
            if token:
                reply = "No pending action was found to confirm."
                self._persist_exchange(user_id, user_name, message, reply)
                return reply
            return None

        if self._is_cancel_message(message):
            self._pending_actions.pop(user_id, None)
            reply = "Understood. I cancelled the pending side-effect actions."
            self._persist_exchange(user_id, user_name, message, reply)
            return reply

        if not token:
            return None

        if token != pending.token:
            reply = (
                "That confirmation token is invalid. "
                f"Reply with `confirm {pending.token}` or `cancel`."
            )
            self._persist_exchange(user_id, user_name, message, reply)
            return reply

        tool_results = await self._execute_tool_calls(pending.tool_calls)
        self._pending_actions.pop(user_id, None)
        reply = self._format_confirmed_results(pending.tool_calls, tool_results)
        self._persist_exchange(user_id, user_name, message, reply)
        return reply

    @staticmethod
    def _assistant_message_from_response(
        response: LLMResponse,
    ) -> dict[str, Any]:
        """Build an assistant message dict suitable for the conversation."""
        msg: dict[str, Any] = {"role": "assistant"}
        if response.text:
            msg["content"] = response.text
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_message(
        self, user_id: int, user_name: str, message: str
    ) -> str:
        """Process a user message through the full agentic loop.

        Returns the final text response from the LLM.
        """
        confirmation_reply = await self._maybe_handle_confirmation(
            user_id, user_name, message
        )
        if confirmation_reply is not None:
            return confirmation_reply

        # Inject live datetime into the system prompt for this turn.
        self._llm.system_prompt = self._inject_datetime(self._system_prompt)

        messages = self._build_messages(user_id, user_name, message)
        tool_defs = self._tools.get_all_schemas()

        response: LLMResponse | None = None
        rounds = 0

        while rounds < _MAX_TOOL_ROUNDS:
            rounds += 1
            response = await self._llm.chat(messages, tools=tool_defs or None)

            # Append assistant turn to the running conversation.
            messages.append(self._assistant_message_from_response(response))

            if not response.tool_calls:
                break

            if any(tc.name in _SENSITIVE_TOOLS for tc in response.tool_calls):
                existing = self._get_pending_action(user_id)
                if existing:
                    final_text = (
                        "You already have a pending confirmation. "
                        f"Reply with `confirm {existing.token}` or `cancel` first."
                    )
                else:
                    final_text = self._stage_pending_action(user_id, response.tool_calls)
                self._persist_exchange(user_id, user_name, message, final_text)
                return final_text

            # Execute tools and feed results back.
            tool_results = await self._execute_tool_calls(response.tool_calls)
            messages.extend(tool_results)

        final_text = (response.text if response else None) or (
            "I do beg your pardon — I seem to have lost my train of thought."
        )

        self._persist_exchange(user_id, user_name, message, final_text)

        return final_text

    async def get_today_summary(self, user_id: int) -> str:
        """Ask the LLM to summarise today's calendar for *user_id*."""
        prompt = (
            "Please check today's calendar and give me a concise summary "
            "of the day's schedule.  If there is nothing scheduled, "
            "let me know."
        )
        return await self.handle_message(user_id, "User", prompt)

    async def get_week_summary(self, user_id: int) -> str:
        """Ask the LLM to summarise the coming week for *user_id*."""
        prompt = (
            "Please check the calendar for the next seven days and give "
            "me a concise overview of the week ahead.  Highlight anything "
            "that needs preparation."
        )
        return await self.handle_message(user_id, "User", prompt)
