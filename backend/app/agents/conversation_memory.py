"""Conversation Memory Agent backed by a small SQLite conversation store."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


class ConversationMemoryAgent:
    """Loads recent turns and persists user/assistant messages per conversation."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path or settings.conversation_db_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(conversation_id, id)"
            )

    def load(self, conversation_id: str) -> list[dict[str, str]]:
        limit = max(2, settings.conversation_history_limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM conversation_messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def save_exchange(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO conversation_messages(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (conversation_id, "user", user_message, timestamp),
                    (conversation_id, "assistant", assistant_message, timestamp),
                ],
            )
            limit = max(2, settings.conversation_history_limit)
            connection.execute(
                """DELETE FROM conversation_messages
                   WHERE conversation_id = ? AND id NOT IN (
                     SELECT id FROM conversation_messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
                   )""",
                (conversation_id, conversation_id, limit),
            )

    def clear(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
