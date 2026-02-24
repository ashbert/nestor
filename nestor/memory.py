"""SQLite-based conversation memory and metadata store for Nestor."""

import sqlite3
from datetime import datetime, timedelta, timezone


class MemoryStore:
    """Persistent storage for conversations, user metadata, and notes."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
                content     TEXT    NOT NULL,
                tool_name   TEXT,
                timestamp   DATETIME DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_user_ts
                ON conversations(user_id, timestamp DESC);

            CREATE TABLE IF NOT EXISTS user_metadata (
                user_id     INTEGER NOT NULL,
                key         TEXT    NOT NULL,
                value       TEXT    NOT NULL,
                updated_at  DATETIME DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, key)
            );

            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  DATETIME DEFAULT (datetime('now')),
                updated_at  DATETIME DEFAULT (datetime('now'))
            );
        """)
        self.conn.commit()

    # ── Conversation messages ────────────────────────────────────────

    def save_message(
        self,
        user_id: int,
        role: str,
        content: str,
        tool_name: str | None = None,
    ) -> None:
        """Persist a single conversation message."""
        self.conn.execute(
            "INSERT INTO conversations (user_id, role, content, tool_name) "
            "VALUES (?, ?, ?, ?)",
            (user_id, role, content, tool_name),
        )
        self.conn.commit()

    def get_recent_messages(
        self, user_id: int, limit: int = 50
    ) -> list[dict]:
        """Return the most recent messages for a user (oldest-first)."""
        rows = self.conn.execute(
            "SELECT role, content, tool_name, timestamp "
            "FROM conversations "
            "WHERE user_id = ? "
            "ORDER BY timestamp DESC, id DESC "
            "LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "tool_name": r["tool_name"],
                "timestamp": r["timestamp"],
            }
            for r in reversed(rows)  # flip back to chronological order
        ]

    def clear_old_messages(self, days: int = 30) -> None:
        """Delete messages older than *days* days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.conn.execute(
            "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
        )
        self.conn.commit()

    # ── User metadata ────────────────────────────────────────────────

    def get_user_meta(self, user_id: int, key: str) -> str | None:
        """Retrieve a single metadata value (or None)."""
        row = self.conn.execute(
            "SELECT value FROM user_metadata WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return row["value"] if row else None

    def set_user_meta(self, user_id: int, key: str, value: str) -> None:
        """Upsert a user metadata key/value pair."""
        self.conn.execute(
            "INSERT INTO user_metadata (user_id, key, value, updated_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(user_id, key) DO UPDATE "
            "SET value = excluded.value, updated_at = excluded.updated_at",
            (user_id, key, value),
        )
        self.conn.commit()

    # ── Notes (Nestor's scratchpad) ──────────────────────────────────

    def save_note(self, title: str, content: str) -> int:
        """Create a new note and return its id."""
        cur = self.conn.execute(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            (title, content),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_notes(self, query: str | None = None) -> list[dict]:
        """List notes, optionally filtering by a substring in title or content."""
        if query:
            pattern = f"%{query}%"
            rows = self.conn.execute(
                "SELECT id, title, content, created_at, updated_at "
                "FROM notes "
                "WHERE title LIKE ? OR content LIKE ? "
                "ORDER BY updated_at DESC",
                (pattern, pattern),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, title, content, created_at, updated_at "
                "FROM notes ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "content": r["content"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
