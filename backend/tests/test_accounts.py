"""Password authentication and cross-account isolation.

These are the tests where a failure means one user reading another's data, or a
stolen database yielding usable passwords, so they assert the boundaries
directly rather than trusting the happy path.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.accounts import (
    AccountStore,
    EmailAlreadyRegistered,
    InvalidCredentials,
    hash_password,
    normalise_email,
    validate_api_key,
    validate_password,
    verify_password,
)
from app.core.conversation import ConversationStore

PASSWORD = "correct-horse-battery"


def _settings(tmp_path) -> Settings:
    return Settings(openai_api_key="test", conversations_db=str(tmp_path / "app.db"))


def _accounts(tmp_path) -> AccountStore:
    return AccountStore(_settings(tmp_path))


# ------------------------------------------------------------------ passwords


def test_password_hash_is_not_the_password():
    digest = hash_password(PASSWORD)

    assert PASSWORD not in digest
    assert digest.startswith("$2b$")


def test_hashes_are_salted_so_equal_passwords_differ():
    # Without a per-password salt, identical passwords share a hash and one
    # cracked entry exposes every account that reused it.
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_verify_accepts_the_right_password_and_rejects_others():
    digest = hash_password(PASSWORD)

    assert verify_password(PASSWORD, digest) is True
    assert verify_password("wrong", digest) is False


def test_verify_survives_a_corrupt_hash():
    assert verify_password(PASSWORD, "not-a-bcrypt-hash") is False


@pytest.mark.parametrize("password", ["", "short", "1234567"])
def test_short_passwords_are_rejected(password):
    with pytest.raises(InvalidCredentials):
        validate_password(password)


def test_overlong_passwords_are_rejected_not_silently_truncated():
    # bcrypt ignores bytes past 72; accepting them would mean two different
    # passwords authenticating the same account.
    with pytest.raises(InvalidCredentials):
        validate_password("a" * 73)


@pytest.mark.parametrize("email", ["", "not-an-email", "a@b", "@example.com", "a b@c.com"])
def test_malformed_emails_are_rejected(email):
    with pytest.raises(InvalidCredentials):
        normalise_email(email)


@pytest.mark.parametrize("key", ["", "hunter2", "sk-", "pk-" + "a" * 40])
def test_malformed_api_keys_are_rejected(key):
    with pytest.raises(InvalidCredentials):
        validate_api_key(key)


# ------------------------------------------------------------- registration


def test_register_then_authenticate(tmp_path):
    store = _accounts(tmp_path)
    created = store.register("user@example.com", PASSWORD)

    signed_in = store.authenticate("user@example.com", PASSWORD)

    assert signed_in.id == created.id


def test_email_is_normalised_at_both_ends(tmp_path):
    store = _accounts(tmp_path)
    created = store.register("  User@Example.COM ", PASSWORD)

    assert store.authenticate("user@example.com", PASSWORD).id == created.id


def test_duplicate_registration_is_refused(tmp_path):
    store = _accounts(tmp_path)
    store.register("user@example.com", PASSWORD)

    with pytest.raises(EmailAlreadyRegistered):
        store.register("user@example.com", "another-password")


def test_wrong_password_is_refused(tmp_path):
    store = _accounts(tmp_path)
    store.register("user@example.com", PASSWORD)

    with pytest.raises(InvalidCredentials):
        store.authenticate("user@example.com", "not-the-password")


def test_unknown_email_and_wrong_password_report_the_same_error(tmp_path):
    # Distinguishing them turns the login form into an account enumerator.
    store = _accounts(tmp_path)
    store.register("user@example.com", PASSWORD)

    with pytest.raises(InvalidCredentials) as unknown:
        store.authenticate("nobody@example.com", PASSWORD)
    with pytest.raises(InvalidCredentials) as wrong:
        store.authenticate("user@example.com", "nope")

    assert str(unknown.value) == str(wrong.value)


def test_accounts_are_distinct_per_email(tmp_path):
    store = _accounts(tmp_path)

    a = store.register("a@example.com", PASSWORD)
    b = store.register("b@example.com", PASSWORD)

    assert a.id != b.id


def test_account_id_is_unrelated_to_the_email(tmp_path):
    # The id is a random uuid, so it leaks nothing and survives an email change.
    account = _accounts(tmp_path).register("user@example.com", PASSWORD)

    assert "user" not in account.id and "@" not in account.id


def test_stored_database_contains_no_plaintext_password(tmp_path):
    settings = _settings(tmp_path)
    AccountStore(settings).register("user@example.com", PASSWORD)

    raw = (tmp_path / "app.db").read_bytes()

    assert PASSWORD.encode() not in raw


def test_password_change_requires_the_current_one(tmp_path):
    store = _accounts(tmp_path)
    account = store.register("user@example.com", PASSWORD)

    with pytest.raises(InvalidCredentials):
        store.change_password(account.id, "wrong-current", "brand-new-password")

    store.change_password(account.id, PASSWORD, "brand-new-password")

    assert store.authenticate("user@example.com", "brand-new-password").id == account.id
    with pytest.raises(InvalidCredentials):
        store.authenticate("user@example.com", PASSWORD)


# ------------------------------------------------------------------ isolation


def _conversations(tmp_path) -> ConversationStore:
    return ConversationStore(_settings(tmp_path))


def test_conversations_are_listed_per_account(tmp_path):
    store = _conversations(tmp_path)
    mine = store.get_or_create(None, account_id="acct-a")
    store.set_title(mine.id, "My question about physics")
    store.get_or_create(None, account_id="acct-b")

    listed = store.list_for_account("acct-a")

    assert [c.id for c in listed] == [mine.id]
    assert listed[0].title == "My question about physics"


def test_another_accounts_conversation_id_is_refused(tmp_path):
    store = _conversations(tmp_path)
    theirs = store.get_or_create(None, account_id="acct-b")

    # Even holding the exact id, a different account cannot open it.
    with pytest.raises(PermissionError):
        store.get_or_create(theirs.id, account_id="acct-a")


def test_peek_reports_the_owning_account(tmp_path):
    # The API's ownership checks read this field. If peek() omits it the value
    # defaults to "", which reads as "unowned" and passes every check — this
    # regressed once and let one account read another's thread.
    store = _conversations(tmp_path)
    theirs = store.get_or_create(None, account_id="acct-b")

    assert store.peek(theirs.id).account_id == "acct-b"


def test_peek_returns_none_for_unknown_ids(tmp_path):
    assert _conversations(tmp_path).peek("does-not-exist") is None


def test_owner_can_reopen_their_own_conversation(tmp_path):
    store = _conversations(tmp_path)
    mine = store.get_or_create(None, account_id="acct-a")
    store.append(mine.id, "user", "hello")

    reopened = store.get_or_create(mine.id, account_id="acct-a")

    assert [t.content for t in reopened.turns] == ["hello"]


def test_titles_are_set_once_from_the_opening_question(tmp_path):
    store = _conversations(tmp_path)
    conversation = store.get_or_create(None, account_id="acct-a")

    store.set_title(conversation.id, "First question")
    store.set_title(conversation.id, "Second question")

    assert store.list_for_account("acct-a")[0].title == "First question"


def test_legacy_conversations_without_an_account_stay_readable(tmp_path):
    # Rows created before accounts existed have account_id ''; they must not
    # become unreachable after the migration.
    store = _conversations(tmp_path)
    legacy = store.get_or_create(None, account_id="")
    store.append(legacy.id, "user", "old message")

    reopened = store.get_or_create(legacy.id, account_id="acct-a")

    assert [t.content for t in reopened.turns] == ["old message"]
