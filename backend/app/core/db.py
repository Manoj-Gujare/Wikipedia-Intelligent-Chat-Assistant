"""One SQLite connection, shared by every store that uses the app database.

Accounts and conversations live in the same file. Giving each store its own
connection made them compete for SQLite's single write lock, which surfaced as
intermittent "database is locked" errors under normal use — two writes landing
close together is not an edge case when signing in also touches conversations.

One connection guarded by one lock removes the contention rather than tuning it:
every statement is short, so serialising them costs nothing measurable and makes
the failure mode impossible.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from threading import Lock


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        # WAL lets reads proceed during a write; the busy timeout covers the
        # remaining case of two writers overlapping.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()


@lru_cache
def get_database(path: str) -> Database:
    return Database(Path(path))
