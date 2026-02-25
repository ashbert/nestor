"""Local long-term memory tools backed by SQLite."""

from __future__ import annotations

from typing import Any

from nestor.memory import MemoryStore
from nestor.tools import BaseTool

__all__ = ["RememberThoughtTool", "RecallThoughtsTool"]


class RememberThoughtTool(BaseTool):
    """Persist a user-provided note for long-term recall."""

    name = "remember_thought"
    description = (
        "Save a long-term private memory note into local SQLite storage for future recall."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory text to store.",
            },
            "title": {
                "type": "string",
                "description": "Optional short title for the memory.",
            },
        },
        "required": ["content"],
    }

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    async def execute(self, **kwargs: Any) -> str:
        content: str = kwargs["content"].strip()
        if not content:
            return "Cannot store an empty memory."
        title: str = (kwargs.get("title") or "Memory").strip() or "Memory"
        note_id = self._memory.save_note(title=title, content=content)
        return f'Memory saved (id={note_id}, title="{title}").'


class RecallThoughtsTool(BaseTool):
    """Recall previously saved long-term notes."""

    name = "recall_thoughts"
    description = (
        "Recall previously saved long-term memory notes, optionally filtered by a query."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional text filter to search within saved memories.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of notes to return (default 10, max 20).",
            },
        },
        "required": [],
    }

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query")
        try:
            limit = int(kwargs.get("limit", 10))
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(limit, 20))

        notes = self._memory.get_notes(query=query)[:limit]
        if not notes:
            if query:
                return f'No saved memories matched "{query}".'
            return "No memories are stored yet."

        lines = [f"Found {len(notes)} memory note(s):"]
        for note in notes:
            content = note["content"].strip()
            if len(content) > 240:
                content = content[:240] + "... [truncated]"
            lines.append(
                f'- [{note["id"]}] {note["title"]} ({note["updated_at"]})\n'
                f"  {content}"
            )
        return "\n".join(lines)
