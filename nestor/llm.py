"""LLM provider abstraction for Nestor.

Supports Anthropic (Claude) and OpenAI with a unified async interface
including tool / function-calling support.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool/function invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalised response from any LLM provider."""

    text: str | None = None
    tool_calls: list[ToolCall] | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider(abc.ABC):
    """Async LLM backend that can chat and optionally call tools."""

    _system_prompt: str

    @property
    def system_prompt(self) -> str:
        """The current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send *messages* to the model and return a normalised response.

        Parameters
        ----------
        messages:
            Conversation turns.  Each dict must contain at least ``role``
            and ``content`` keys.  Tool-result messages use the format
            native to the provider.
        tools:
            Optional list of tool definitions (JSON-Schema style).
        """


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


async def _retry(coro_factory, *, retries: int = _MAX_RETRIES):
    """Call *coro_factory()* up to *retries* times with exponential backoff.

    *coro_factory* must be a zero-arg callable that returns a new awaitable
    each time (because a spent coroutine cannot be re-awaited).
    """
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except (
            anthropic.RateLimitError,
            openai.RateLimitError,
            anthropic.APIStatusError,
            openai.APIStatusError,
        ) as exc:
            last_exc = exc
            if attempt == retries:
                break
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the native Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert generic tool defs to Anthropic's format.

        Accepts either Anthropic-native dicts (with ``input_schema``) or
        OpenAI-style dicts (``{type, function: {name, description, parameters}}``)
        and normalises to the Anthropic shape.
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if "input_schema" in tool:
                # Already Anthropic-native.
                converted.append(tool)
            elif "function" in tool:
                func = tool["function"]
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            else:
                converted.append(tool)
        return converted

    @staticmethod
    def _parse_response(response: anthropic.types.Message) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )

    # -- public -----------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = await _retry(lambda: self._client.messages.create(**kwargs))
        return self._parse_response(response)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the Chat Completions API with function calling."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure every tool dict is in OpenAI's ``{type, function}`` shape."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if "function" in tool:
                converted.append(tool)
            else:
                # Anthropic-style → OpenAI-style.
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get(
                                "input_schema",
                                tool.get("parameters", {"type": "object", "properties": {}}),
                            ),
                        },
                    }
                )
        return converted

    def _build_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepend the system prompt if set."""
        out: list[dict[str, Any]] = []
        if self._system_prompt:
            out.append({"role": "system", "content": self._system_prompt})
        out.extend(messages)
        return out

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        text = message.content

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls or None,
        )

    # -- public -----------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        full_messages = self._build_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": full_messages,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = await _retry(
            lambda: self._client.chat.completions.create(**kwargs)
        )
        return self._parse_response(response)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str = "",
    max_tokens: int = 4096,
) -> LLMProvider:
    """Instantiate an :class:`LLMProvider` by name.

    Parameters
    ----------
    provider:
        ``"anthropic"`` or ``"openai"`` (case-insensitive).
    api_key:
        API key for the chosen provider.
    model:
        Model identifier (e.g. ``"claude-sonnet-4-20250514"``, ``"gpt-4o"``).
    system_prompt:
        Optional system prompt injected into every conversation.
    max_tokens:
        Maximum tokens for the model response.
    """
    name = provider.strip().lower()
    if name == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    if name == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}  (expected 'anthropic' or 'openai')")
