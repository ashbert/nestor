"""Tool registry system for Nestor."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["BaseTool", "ToolRegistry"]


class BaseTool(ABC):
    """Base class for all tools available to Nestor.

    Subclasses must set *name*, *description*, and *parameters* and implement
    the async :meth:`execute` method.
    """

    name: str
    """Unique identifier used to invoke the tool."""

    description: str
    """Human-/LLM-readable description of what the tool does."""

    parameters: dict[str, Any]
    """JSON Schema describing the tool's accepted parameters.

    Expected shape::

        {
            "type": "object",
            "properties": { ... },
            "required": [ ... ],
        }
    """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a result string.

        Implementations should catch foreseeable errors and return a
        human-readable error message rather than raising, so that the LLM can
        react gracefully.
        """

    # Convenience -----------------------------------------------------------------

    def schema(self) -> dict[str, Any]:
        """Return the tool schema in the generic format expected by LLMs."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of tools available to Nestor.

    Usage::

        registry = ToolRegistry()
        registry.register(MyTool())
        schemas = registry.get_all_schemas()
        result = await registry.execute("my_tool", {"arg": "value"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # Mutation --------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool.  Raises *ValueError* on duplicate names."""
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered."
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool %r", tool.name)

    # Lookup ----------------------------------------------------------------------

    def get_tool(self, name: str) -> BaseTool:
        """Return the tool registered under *name*.

        Raises *KeyError* if no such tool exists.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"No tool registered with name {name!r}") from None

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-serialisable schemas for every registered tool."""
        return [tool.schema() for tool in self._tools.values()]

    # Execution -------------------------------------------------------------------

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Look up *name* and execute with *arguments*.

        Returns the tool's result string.  If the tool is not found, an error
        string is returned instead of raising so the LLM can recover.
        """
        try:
            tool = self.get_tool(name)
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            return f"Error: unknown tool {name!r}. Available tools: {available}"

        try:
            return await tool.execute(**arguments)
        except Exception:
            logger.exception("Unhandled error executing tool %r", name)
            return f"Error: an unexpected error occurred while running {name!r}."

    # Helpers ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._tools))
        return f"ToolRegistry([{names}])"
