"""Nestor's brain — orchestrates LLM, tools, and memory.

The :class:`NestorBrain` runs the agentic loop: it loads conversation
history, calls the LLM, executes any requested tools, feeds results
back, and persists the exchange.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
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
    "website",
    "web",
    "source",
    "official source",
    "when does",
}

_CALENDAR_TROUBLESHOOT_HINTS = {
    "could not find this event",
    "couldn't find this event",
    "cant find this event",
    "can't find this event",
    "cannot find this event",
    "not seeing this event",
    "don't see this event",
    "dont see this event",
    "where is this event",
}

_ACTION_INTENT_KEYWORDS = {
    " add ",
    " create ",
    " schedule ",
    " reschedule ",
    " move ",
    " delete ",
    " remove ",
    " cancel ",
    " update ",
}

_ACTION_TARGET_KEYWORDS = {
    " calendar",
    " event",
    " appointment",
    " meeting",
    " reminder",
    " recurring",
    " every week",
}
_DEEP_KEYWORDS = {
    "deep research",
    "travel plan",
    "itinerary",
    "summer camp",
    "camp planning",
    "pantry",
    "fridge",
    "shopping list",
    "dinner",
    "analyze",
    "analysis",
    "compare options",
    "compare",
    "opinion on previous conversations",
    "previous conversations",
    "multi model",
}
_PARALLEL_RESEARCH_HINTS = {
    "deep research",
    "multi source",
    "compare sources",
    "travel",
    "camp",
    "school calendar",
}
_URL_RE = re.compile(r"https?://[^\s)>\"]+")
_CALENDAR_DATE_HINT_RE = re.compile(
    r"\b("
    r"today|tomorrow|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b"
)
_CALENDAR_TIME_HINT_RE = re.compile(
    r"\b("
    r"\d{1,2}:\d{2}\s*(?:am|pm)?|"
    r"\d{1,2}\s*(?:am|pm)|"
    r"between\s+\d"
    r")\b"
)
_DAY_NAME_TO_WEEKDAY = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
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
        llm_fast: LLMProvider | None = None,
        llm_deep: LLMProvider | None = None,
        channel_model_overrides: dict[str, str] | None = None,
        enable_parallel_research: bool = True,
    ) -> None:
        self._llm = llm
        self._llm_fast = llm_fast or llm
        self._llm_deep = llm_deep or llm
        self._tools = tool_registry
        self._memory = memory
        self._system_prompt = system_prompt
        self._channel_model_overrides = channel_model_overrides or {}
        self._enable_parallel_research = enable_parallel_research
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
        # "find" in troubleshooting follow-ups like "could not find this event"
        # should not force the research workflow.
        if any(hint in text for hint in _CALENDAR_TROUBLESHOOT_HINTS):
            return False
        return any(keyword in text for keyword in _RESEARCH_KEYWORDS)

    @staticmethod
    def _looks_like_action_request(message: str) -> bool:
        text = f" {re.sub(r'\s+', ' ', message.lower()).strip()} "
        if not text.strip():
            return False
        has_action_intent = any(keyword in text for keyword in _ACTION_INTENT_KEYWORDS)
        if not has_action_intent:
            return False
        return any(keyword in text for keyword in _ACTION_TARGET_KEYWORDS)

    @staticmethod
    def _looks_like_calendar_create_request(message: str) -> bool:
        text = re.sub(r"\s+", " ", message.lower()).strip()
        if not text:
            return False

        action_terms = (
            "calendar",
            "add",
            "schedule",
            "create",
            "appointment",
            "meeting",
            "event",
        )
        has_action = any(term in text for term in action_terms)
        if not has_action:
            return False

        has_date_or_time = bool(
            _CALENDAR_DATE_HINT_RE.search(text) or _CALENDAR_TIME_HINT_RE.search(text)
        )
        return has_date_or_time

    @staticmethod
    def _is_calendar_troubleshooting_followup(message: str) -> bool:
        text = re.sub(r"\s+", " ", message.lower()).strip()
        if not text:
            return False
        return any(hint in text for hint in _CALENDAR_TROUBLESHOOT_HINTS)

    def _resolve_date_hint(self, message: str) -> str | None:
        text = re.sub(r"\s+", " ", message.lower()).strip()
        if not text:
            return None

        tz_name = os.environ.get("NESTOR_TIMEZONE", "America/Los_Angeles")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("America/Los_Angeles")
        today = datetime.now(tz=tz).date()

        if "today" in text:
            return today.isoformat()
        if "tomorrow" in text:
            return (today + timedelta(days=1)).isoformat()

        for name, weekday in _DAY_NAME_TO_WEEKDAY.items():
            if name not in text:
                continue
            days_ahead = (weekday - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()

        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if iso_match:
            return iso_match.group(1)

        md_match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
        if md_match:
            month = int(md_match.group(1))
            day = int(md_match.group(2))
            year_part = md_match.group(3)
            year = today.year
            if year_part:
                year = int(year_part)
                if year < 100:
                    year += 2000
            try:
                parsed = date(year, month, day)
            except ValueError:
                return None
            if not year_part and parsed < today:
                try:
                    parsed = date(year + 1, month, day)
                except ValueError:
                    return None
            return parsed.isoformat()

        return None

    @staticmethod
    def _extract_calendar_search_terms(message: str) -> list[str]:
        text = re.sub(r"[^a-z0-9\s]", " ", message.lower())
        tokens = [t for t in text.split() if t]
        stop = {
            "could", "couldnt", "couldn", "not", "find", "this", "event",
            "cant", "can", "t", "cannot", "where", "is", "the", "a", "an",
            "i", "see", "don", "dont", "do", "no", "on", "for", "at", "to",
            "my", "calendar", "it", "that", "was", "be", "please",
        }
        terms: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in stop or tok.isdigit() or len(tok) < 3:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            terms.append(tok)
        return terms[:4]

    @staticmethod
    def _looks_like_deep_request(message: str) -> bool:
        text = re.sub(r"\s+", " ", message.lower()).strip()
        if not text:
            return False
        if any(keyword in text for keyword in _DEEP_KEYWORDS):
            return True
        return len(text) >= 260

    @staticmethod
    def _wants_parallel_research(message: str) -> bool:
        text = re.sub(r"\s+", " ", message.lower()).strip()
        if not text:
            return False
        return any(keyword in text for keyword in _PARALLEL_RESEARCH_HINTS)

    def _provider_label(self, provider: LLMProvider) -> str:
        if provider is self._llm_fast:
            return "fast"
        if provider is self._llm_deep:
            return "deep"
        return "default"

    def _select_llm(
        self, message: str, context: dict[str, str] | None
    ) -> LLMProvider:
        channel_id = (context or {}).get("channel_id", "")
        override = self._channel_model_overrides.get(channel_id, "").strip().lower()
        if override in {"fast", "cheap", "quick"}:
            return self._llm_fast
        if override in {"deep", "quality", "research"}:
            return self._llm_deep

        if self._looks_like_deep_request(message):
            return self._llm_deep

        if self._looks_like_research_request(message):
            return self._llm_fast

        return self._llm_fast

    @staticmethod
    def _derive_research_queries(message: str) -> list[str]:
        text = re.sub(r"\s+", " ", message).strip()
        if not text:
            return []
        queries = [text]
        lower = text.lower()
        if "school" in lower and "calendar" in lower:
            queries.append(f"{text} official district calendar")
            queries.append(f"{text} pdf")
        elif "travel" in lower:
            queries.append(f"{text} official tourism board")
            queries.append(f"{text} family itinerary ideas")
        else:
            queries.append(f"{text} official source")
            queries.append(f"{text} latest update")

        deduped: list[str] = []
        seen: set[str] = set()
        for q in queries:
            qn = q.strip()
            if not qn:
                continue
            key = qn.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(qn)
        return deduped[:3]

    @staticmethod
    def _extract_urls(text: str, *, limit: int = 4) -> list[str]:
        if not text:
            return []
        urls: list[str] = []
        seen: set[str] = set()
        for match in _URL_RE.findall(text):
            url = match.rstrip(".,;")
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                continue
            if not parsed.netloc:
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append(url)
            if len(urls) >= limit:
                break
        return urls

    async def _build_parallel_research_brief(self, message: str) -> str | None:
        if "web_search" not in self._tools or "fetch_web_page" not in self._tools:
            return None

        queries = self._derive_research_queries(message)
        if not queries:
            return None

        search_calls = [
            self._tools.execute(
                "web_search",
                {"query": query, "num_results": 6},
            )
            for query in queries
        ]
        search_results = await asyncio.gather(*search_calls, return_exceptions=True)

        candidate_urls: list[str] = []
        lines = ["Parallel research workers gathered the following:"]
        for query, result in zip(queries, search_results):
            if isinstance(result, Exception):
                lines.append(f"- Search error for '{query}': {result}")
                continue
            result_text = str(result)
            lines.append(f"- Query: {query}")
            urls = self._extract_urls(result_text, limit=3)
            candidate_urls.extend(urls)
            for url in urls:
                lines.append(f"  - {url}")

        unique_urls: list[str] = []
        seen_urls: set[str] = set()
        for url in candidate_urls:
            key = url.lower()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            unique_urls.append(url)
        unique_urls = unique_urls[:4]

        if unique_urls:
            fetch_calls = [
                self._tools.execute("fetch_web_page", {"url": url})
                for url in unique_urls
            ]
            fetch_results = await asyncio.gather(*fetch_calls, return_exceptions=True)
            for url, fetched in zip(unique_urls, fetch_results):
                if isinstance(fetched, Exception):
                    lines.append(f"- Fetch error: {url} ({fetched})")
                    continue
                text = str(fetched).strip()
                if len(text) > 900:
                    text = text[:900] + "\n[...truncated]"
                lines.append(f"- Fetched: {url}\n{text}")

        if len(lines) <= 1:
            return None
        return "\n".join(lines)

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

    async def _maybe_handle_calendar_troubleshooting(
        self,
        user_id: int,
        user_name: str,
        message: str,
    ) -> str | None:
        if not self._is_calendar_troubleshooting_followup(message):
            return None

        date_hint = self._resolve_date_hint(message)

        # Prefer direct date listing when a date/day hint is present.
        if date_hint and "list_calendar_events" in self._tools:
            listed = await self._tools.execute(
                "list_calendar_events",
                {"start_date": date_hint, "end_date": date_hint},
            )
            listed_text = str(listed)
            if listed_text.startswith("Found "):
                reply = (
                    "Certainly. I checked the calendar directly. "
                    f"Here are the entries for {date_hint}:\n\n{listed_text}"
                )
            else:
                reply = (
                    "I checked the calendar directly and found no events on "
                    f"{date_hint}. If you wish, I can recreate it now."
                )
            self._persist_exchange(user_id, user_name, message, reply)
            return reply

        if "search_calendar_events" not in self._tools:
            return None

        search_terms = self._extract_calendar_search_terms(message)
        if not search_terms:
            search_terms = ["appointment", "meeting", "event"]

        for query in search_terms:
            result = await self._tools.execute(
                "search_calendar_events", {"query": query, "days_ahead": 30}
            )
            text = str(result)
            if text.startswith("Found "):
                reply = (
                    "Certainly. I checked the calendar directly. "
                    "Here is the closest matching result:\n\n"
                    f"{text}"
                )
                self._persist_exchange(user_id, user_name, message, reply)
                return reply

        reply = (
            "I checked the calendar directly and do not see a matching event. "
            "If you wish, I can recreate it now."
        )
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
        self,
        user_id: int,
        user_name: str,
        message: str,
        context: dict[str, str] | None = None,
    ) -> str:
        """Process a user message through the full agentic loop.

        Returns the final text response from the LLM.
        """
        confirmation_reply = await self._maybe_handle_confirmation(
            user_id, user_name, message
        )
        if confirmation_reply is not None:
            return confirmation_reply

        calendar_troubleshooting_reply = await self._maybe_handle_calendar_troubleshooting(
            user_id, user_name, message
        )
        if calendar_troubleshooting_reply is not None:
            return calendar_troubleshooting_reply

        llm = self._select_llm(message, context)
        llm.system_prompt = self._inject_datetime(self._system_prompt)
        logger.info(
            "Selected model tier for message: %s (source=%s, channel=%s)",
            self._provider_label(llm),
            (context or {}).get("source", "default"),
            (context or {}).get("channel_id", ""),
        )

        messages = self._build_messages(user_id, user_name, message)
        tool_defs = self._tools.get_all_schemas()
        action_request = self._looks_like_action_request(message)
        calendar_action_request = self._looks_like_calendar_create_request(message)
        research_request = self._looks_like_research_request(message) and not action_request
        if action_request and self._looks_like_research_request(message):
            logger.info("Research forcing suppressed due to action-oriented request")
        escalated_to_deep = False
        executed_tool_names: set[str] = set()

        if self._enable_parallel_research and self._wants_parallel_research(message):
            try:
                brief = await self._build_parallel_research_brief(message)
            except Exception:
                logger.exception("Parallel research prefetch failed")
                brief = None
            if brief:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reference dossier from internal research workers:\n"
                            f"{brief}\n\n"
                            "Use this as evidence, verify with tools as needed, and cite sources."
                        ),
                    }
                )

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
            if research_request and rounds > 1:
                # Once a non-research action tool runs (e.g., calendar create),
                # avoid drifting into web research in later rounds.
                force_tool_use = False
            # On the last round, don't offer tools so the LLM must produce text.
            last_round = rounds == _MAX_TOOL_ROUNDS
            response = await llm.chat(
                messages,
                tools=(tool_defs or None) if not last_round else None,
                force_tool_use=force_tool_use if not last_round else False,
            )

            # Append assistant turn to the running conversation.
            messages.append(self._assistant_message_from_response(response))

            if not response.tool_calls:
                if (
                    research_request
                    and not escalated_to_deep
                    and llm is self._llm_fast
                    and self._llm_deep is not self._llm_fast
                    and rounds < _MAX_TOOL_ROUNDS - 1
                    and not (
                        calendar_action_request
                        and "create_calendar_event" in executed_tool_names
                    )
                ):
                    escalated_to_deep = True
                    llm = self._llm_deep
                    llm.system_prompt = self._inject_datetime(self._system_prompt)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Escalate this request to a deeper reasoning pass. "
                                "Use tools where needed and provide source-backed conclusions."
                            ),
                        }
                    )
                    logger.info("Escalated conversation to deep model tier")
                    continue
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
            executed_tool_names.update(tc.name for tc in response.tool_calls)
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
