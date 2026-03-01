"""Nestor's brain — orchestrates LLM, tools, and memory.

The :class:`NestorBrain` runs the agentic loop: it loads conversation
history, calls the LLM, executes any requested tools, feeds results
back, and persists the exchange.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from nestor.llm import LLMProvider, LLMResponse, ToolCall
from nestor.memory import MemoryStore
from nestor.tools import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 8
_CONFIRMATION_TTL = timedelta(minutes=15)
_RESEARCH_KEYWORDS = {
    "look up",
    "lookup",
    "research",
    "search",
    "find",
    "check",
    "calendar",
    "website",
    "web",
    "source",
    "when does",
    "date",
    "schedule",
    "school",
}

# Irreversible actions that require a simple yes/no confirmation
_CONFIRM_TOOLS = {
    "send_email",
    "delete_calendar_event",
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
        # Pending actions are now persisted in SQLite via self._memory

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
        history_rows = self._memory.get_recent_messages(user_id, limit=30)
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

    @staticmethod
    def _looks_like_research_request(message: str) -> bool:
        text = re.sub(r"\s+", " ", message.lower()).strip()
        if not text:
            return False
        return any(keyword in text for keyword in _RESEARCH_KEYWORDS)

    def _persist_exchange(
        self, user_id: int, user_name: str, user_message: str, assistant_message: str
    ) -> None:
        self._memory.save_message(user_id, "user", f"[{user_name}]: {user_message}")
        self._memory.save_message(user_id, "assistant", assistant_message)

    def _get_pending_action(self, user_id: int) -> PendingAction | None:
        row = self._memory.get_pending_action(user_id)
        if not row:
            return None
        created = datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - created > _CONFIRMATION_TTL:
            self._memory.delete_pending_action(user_id)
            return None
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in json.loads(row["tool_calls"])
        ]
        return PendingAction(token=row["token"], tool_calls=tool_calls, created_at=created)

    @staticmethod
    def _is_yes_message(message: str) -> bool:
        return message.strip().lower() in {
            "yes", "y", "yep", "yeah", "sure", "ok", "do it",
            "go ahead", "proceed", "confirm", "approved", "send it",
        }

    @staticmethod
    def _is_no_message(message: str) -> bool:
        return message.strip().lower() in {
            "no", "n", "nope", "cancel", "deny", "reject", "abort",
            "don't", "dont", "stop", "nevermind", "never mind",
        }

    def _stage_pending_action(self, user_id: int, tool_calls: list[ToolCall]) -> str:
        tc_json = json.dumps([{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls])
        self._memory.save_pending_action(user_id, "yes", tc_json)

        lines = ["Before I proceed, here's what I'm about to do:"]
        for tc in tool_calls:
            if tc.name == "send_email":
                to = tc.arguments.get("to", "?")
                subj = tc.arguments.get("subject", "?")
                lines.append(f"📧 Send email to **{to}** — \"{subj}\"")
            elif tc.name == "delete_calendar_event":
                eid = tc.arguments.get("event_id", tc.arguments.get("query", "?"))
                lines.append(f"🗑️ Delete calendar event: {eid}")
            else:
                args = json.dumps(tc.arguments, ensure_ascii=False)
                if len(args) > 200:
                    args = args[:200] + "…"
                lines.append(f"• `{tc.name}`: {args}")
        lines.append("")
        lines.append("**Yes** or **No**?")
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
        if not pending:
            return None

        if self._is_yes_message(message):
            tool_results = await self._execute_tool_calls(pending.tool_calls)
            self._memory.delete_pending_action(user_id)
            reply = self._format_confirmed_results(pending.tool_calls, tool_results)
            self._persist_exchange(user_id, user_name, message, reply)
            return reply

        if self._is_no_message(message):
            self._memory.delete_pending_action(user_id)
            reply = "Very good, Sir. Consider it banished to the void — along with the Captain's last attempt at soufflé."
            self._persist_exchange(user_id, user_name, message, reply)
            return reply

        # Not a yes/no — clear pending and process as a new message
        self._memory.delete_pending_action(user_id)
        return None

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
        research_request = self._looks_like_research_request(message)

        response: LLMResponse | None = None
        rounds = 0
        retried_with_tool_nudge = False

        while rounds < _MAX_TOOL_ROUNDS:
            rounds += 1
            force_tool_use = bool(
                research_request
                and tool_defs
                and rounds == 1
            )
            # On the last round, don't offer tools so the LLM must produce text.
            last_round = rounds == _MAX_TOOL_ROUNDS
            response = await self._llm.chat(
                messages,
                tools=(tool_defs or None) if not last_round else None,
                force_tool_use=force_tool_use if not last_round else False,
            )

            # Append assistant turn to the running conversation.
            messages.append(self._assistant_message_from_response(response))

            if not response.tool_calls:
                if (
                    research_request
                    and tool_defs
                    and not retried_with_tool_nudge
                    and rounds < _MAX_TOOL_ROUNDS - 1
                ):
                    retried_with_tool_nudge = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please use the available research tools before answering. "
                                "Prefer official sources and provide exact dates when available."
                            ),
                        }
                    )
                    continue
                break

            # Check if any tool calls need confirmation
            needs_confirm = [tc for tc in response.tool_calls if tc.name in _CONFIRM_TOOLS]
            if needs_confirm:
                # Stage all tool calls (confirmed ones gate the batch)
                final_text = self._stage_pending_action(user_id, response.tool_calls)
                self._persist_exchange(user_id, user_name, message, final_text)
                return final_text

            # Execute tools and feed results back.
            tool_results = await self._execute_tool_calls(response.tool_calls)
            messages.extend(tool_results)

        final_text = response.text if response else None
        if not final_text:
            logger.warning(
                "Agentic loop finished after %d rounds with no text response "
                "(user_id=%s, message=%r)",
                rounds,
                user_id,
                message[:100],
            )
            final_text = (
                "I do beg your pardon — I appear to have wandered into "
                "the mental equivalent of a broom cupboard. Might I "
                "trouble you to rephrase the request?"
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
