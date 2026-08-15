"""Accounts: email + password identity.

The account id is a random uuid, decoupled from both the email and the OpenAI
key. That matters:

* the API key becomes a **runtime credential**, not part of who you are — a user
  can rotate their key without losing their conversations or knowledge base;
* the email can be corrected without orphaning data;
* nothing spendable is stored. The password is bcrypt-hashed and the API key is
  never persisted at all.

Passwords are hashed with bcrypt (work factor 12), which is deliberately slow:
a fast hash like SHA-256 is trivially brute-forced from a stolen table.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

import bcrypt

from ..config import Settings, get_settings
from .db import get_database

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
# bcrypt silently truncates beyond 72 bytes, so reject rather than mislead.
MAX_PASSWORD_BYTES = 72
BCRYPT_ROUNDS = 12

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
"""


class InvalidCredentials(ValueError):
    """Raised when credentials are malformed or do not authenticate."""


class EmailAlreadyRegistered(ValueError):
    """Raised when registering an email that already has an account."""


@dataclass(slots=True)
class Account:
    id: str
    email: str
    created_at: float
    last_seen_at: float


def normalise_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise InvalidCredentials("Enter a valid email address.")
    return email


def validate_password(password: str) -> str:
    password = password or ""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise InvalidCredentials(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise InvalidCredentials("Password is too long (72 bytes maximum).")
    return password


def validate_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    # Shape check only — whether the key works is settled by the first OpenAI
    # call, and it is never stored to find out.
    if not key.startswith("sk-") or len(key) < 20:
        raise InvalidCredentials("That doesn't look like an OpenAI API key.")
    return key


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


class AccountStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        database = get_database(str(self.settings.conversations_db_path))
        # Shared connection with the conversation store: same file, one write
        # lock, so the two never contend.
        self._lock = database.lock
        self._db = database.conn
        with self._lock:
            self._migrate()
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def _migrate(self) -> None:
        """Move aside a pre-password `accounts` table.

        Earlier releases identified an account by hashing email + API key and
        stored no password. Those rows cannot be carried forward — there is no
        password to migrate, and the id scheme itself changed — but dropping
        them would destroy data silently. They are renamed instead, so the file
        still holds them if anyone needs to look.
        """
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(accounts)")}
        if not columns or "password_hash" in columns:
            return

        suffix = int(time.time())
        self._db.execute(f"ALTER TABLE accounts RENAME TO accounts_legacy_{suffix}")
        self._db.commit()
        logger.warning(
            "Existing accounts predate password sign-in and were moved to "
            "accounts_legacy_%s. Those users must register again, and any "
            "knowledge-base articles they added are no longer reachable.",
            suffix,
        )

    # ------------------------------------------------------------- lifecycle

    def register(self, email: str, password: str) -> Account:
        email = normalise_email(email)
        validate_password(password)
        now = time.time()
        account_id = uuid.uuid4().hex

        with self._lock:
            existing = self._db.execute(
                "SELECT account_id FROM accounts WHERE email = ?", (email,)
            ).fetchone()
            if existing:
                raise EmailAlreadyRegistered("That email is already registered.")

            self._db.execute(
                "INSERT INTO accounts (account_id, email, password_hash, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (account_id, email, hash_password(password), now, now),
            )
            self._db.commit()

        logger.info("Registered account for %s", _mask(email))
        return Account(id=account_id, email=email, created_at=now, last_seen_at=now)

    def authenticate(self, email: str, password: str) -> Account:
        email = normalise_email(email)

        with self._lock:
            row = self._db.execute(
                "SELECT account_id, password_hash, created_at FROM accounts WHERE email = ?",
                (email,),
            ).fetchone()

        # Same error whether the email is unknown or the password is wrong:
        # distinguishing them turns the login form into an account enumerator.
        # The dummy verify keeps timing similar for unknown emails.
        if row is None:
            verify_password(password, _DUMMY_HASH)
            raise InvalidCredentials("Email or password is incorrect.")

        account_id, password_hash, created_at = row
        if not verify_password(password, password_hash):
            raise InvalidCredentials("Email or password is incorrect.")

        now = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE accounts SET last_seen_at = ? WHERE account_id = ?",
                (now, account_id),
            )
            self._db.commit()

        return Account(id=account_id, email=email, created_at=created_at, last_seen_at=now)

    def change_password(self, account_id: str, current: str, new: str) -> None:
        validate_password(new)
        with self._lock:
            row = self._db.execute(
                "SELECT password_hash FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        if row is None or not verify_password(current, row[0]):
            raise InvalidCredentials("Current password is incorrect.")

        with self._lock:
            self._db.execute(
                "UPDATE accounts SET password_hash = ? WHERE account_id = ?",
                (hash_password(new), account_id),
            )
            self._db.commit()

    # ------------------------------------------------------------------ read

    def get(self, account_id: str) -> Account | None:
        with self._lock:
            row = self._db.execute(
                "SELECT email, created_at, last_seen_at FROM accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if not row:
            return None
        return Account(id=account_id, email=row[0], created_at=row[1], last_seen_at=row[2])

    def close(self) -> None:
        # The connection is shared; closing is the database's job, not ours.
        pass


# Compared against when the email is unknown, so a failed login costs roughly
# the same time either way and cannot be used to probe for registered emails.
_DUMMY_HASH = hash_password("not-a-real-password")


def _mask(email: str) -> str:
    """Log-safe rendering: never write a full address to the log."""
    name, _, domain = email.partition("@")
    head = name[:2] if len(name) > 2 else name[:1]
    return f"{head}***@{domain}"


@lru_cache
def get_account_store() -> AccountStore:
    return AccountStore()
