"""Multi-turn conversation memory, persisted in SQLite.

SQLite rather than the vector store, deliberately: history is ordered,
exact-lookup data — "the last 12 turns of thread X, in order" — which is what a
relational table does natively and a similarity index does not. It survives
backend restarts, WAL mode keeps it safe across multiple workers on one
machine, and it adds zero dependencies. Scaling to a fleet means reimplementing
this class's five methods against Redis or Postgres; nothing else changes.

Assistant turns can carry a ``meta`` JSON blob (cited sources, article links)
so a restored conversation keeps its citations, not just its text.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..config import Settings, get_settings
from .db import get_database

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class Conversation:
    id: str
    turns: list[Turn] = field(default_factory=list)
    lang: str = "en"
    updated_at: float = field(default_factory=time.time)
    account_id: str = ""
    title: str = ""


# Tables only. Indexes are created *after* `_migrate()` adds any columns
# introduced by a later release — otherwise opening an older database fails on
# an index that references a column the table does not have yet.
_SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    lang TEXT NOT NULL DEFAULT 'en',
    updated_at REAL NOT NULL,
    account_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS turns (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT,
    created_at REAL NOT NULL
);
"""

_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_conversations_account
    ON conversations(account_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id, seq);
"""


class ConversationStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # One connection shared with the account store, guarded by one lock:
        # every operation here is a handful of sub-millisecond statements, so
        # serialising them costs nothing and rules out lock contention.
        database = get_database(str(self.settings.conversations_db_path))
        self._lock = database.lock
        self._db = database.conn
        with self._lock:
            self._db.executescript(_SCHEMA_TABLES)
            self._migrate()
            self._db.executescript(_SCHEMA_INDEXES)
            self._db.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the first release, in place.

        A database created before accounts existed has no `account_id`/`title`;
        adding them here keeps existing conversations readable instead of
        forcing the file to be deleted.
        """
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(conversations)")}
        for column, ddl in (
            ("account_id", "ALTER TABLE conversations ADD COLUMN account_id TEXT NOT NULL DEFAULT ''"),
            ("title", "ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in existing:
                self._db.execute(ddl)
                logger.info("Migrated conversations table: added %s", column)

    def close(self) -> None:
        # The connection is shared; closing is the database's job, not ours.
        pass

    # ------------------------------------------------------------------ reads

    def _turns(self, conversation_id: str) -> list[Turn]:
        rows = self._db.execute(
            "SELECT role, content, meta, created_at FROM turns "
            "WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        ).fetchall()
        return [
            Turn(
                role=role,
                content=content,
                timestamp=created_at,
                meta=json.loads(meta) if meta else None,
            )
            for role, content, meta, created_at in rows
        ]

    def history(self, conversation_id: str) -> list[Turn]:
        with self._lock:
            return self._turns(conversation_id)

    def peek(self, conversation_id: str) -> Conversation | None:
        """The conversation as stored, or None — never creates.

        `account_id` must be returned: callers use it to decide whether the
        requester owns this thread, and an empty value would read as
        "unowned" and pass every ownership check.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT lang, updated_at, account_id, title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return Conversation(
                id=conversation_id,
                turns=self._turns(conversation_id),
                lang=row[0],
                updated_at=row[1],
                account_id=row[2],
                title=row[3],
            )

    # ----------------------------------------------------------------- writes

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.settings.conversation_ttl_seconds
        self._db.execute(
            "DELETE FROM turns WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE updated_at < ?)",
            (cutoff,),
        )
        self._db.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,))

    def get_or_create(
        self, conversation_id: str | None, lang: str = "en", account_id: str = ""
    ) -> Conversation:
        with self._lock:
            self._evict_expired()
            if conversation_id:
                row = self._db.execute(
                    "SELECT account_id FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if row:
                    # A conversation belongs to the account that created it.
                    # Refusing to serve someone else's id is what stops a
                    # guessed or leaked id exposing another user's history.
                    if row[0] and account_id and row[0] != account_id:
                        raise PermissionError("conversation belongs to another account")
                    self._db.execute(
                        "UPDATE conversations SET lang = ? WHERE id = ?",
                        (lang, conversation_id),
                    )
                    self._db.commit()
                    return Conversation(
                        id=conversation_id,
                        turns=self._turns(conversation_id),
                        lang=lang,
                        account_id=row[0],
                    )

            new_id = conversation_id or uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO conversations (id, lang, updated_at, account_id) "
                "VALUES (?, ?, ?, ?)",
                (new_id, lang, time.time(), account_id),
            )
            self._db.commit()
            return Conversation(id=new_id, lang=lang, account_id=account_id)

    def list_for_account(self, account_id: str, limit: int = 50) -> list[Conversation]:
        """Newest-first conversation list for the sidebar."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, lang, updated_at, title FROM conversations "
                "WHERE account_id = ? ORDER BY updated_at DESC LIMIT ?",
                (account_id, limit),
            ).fetchall()
        return [
            Conversation(
                id=r[0], lang=r[1], updated_at=r[2], title=r[3], account_id=account_id
            )
            for r in rows
        ]

    def set_title(self, conversation_id: str, title: str) -> None:
        """Label a thread with its opening question, for the sidebar."""
        clean = " ".join(title.split())[:80]
        with self._lock:
            self._db.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = ''",
                (clean, conversation_id),
            )
            self._db.commit()

    def append(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            now = time.time()
            touched = self._db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            if touched.rowcount == 0:
                # Unknown conversation: mirror the old in-memory behaviour and
                # drop the write rather than resurrecting evicted threads.
                return
            self._db.execute(
                "INSERT INTO turns (conversation_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(meta) if meta else None, now),
            )
            # Keep only the newest N turns so prompts stay bounded.
            self._db.execute(
                "DELETE FROM turns WHERE conversation_id = ? AND seq NOT IN ("
                "  SELECT seq FROM turns WHERE conversation_id = ?"
                "  ORDER BY seq DESC LIMIT ?)",
                (conversation_id, conversation_id, self.settings.conversation_max_turns),
            )
            self._db.commit()

    def clear(self, conversation_id: str) -> bool:
        with self._lock:
            self._db.execute(
                "DELETE FROM turns WHERE conversation_id = ?", (conversation_id,)
            )
            deleted = self._db.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            self._db.commit()
            return deleted.rowcount > 0


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()
