"""Conversation persistence: history, trimming and eviction."""

from __future__ import annotations

import time

from app.config import Settings
from app.core.conversation import ConversationStore



def _store(tmp_path, **overrides) -> ConversationStore:
    settings = Settings(
        openai_api_key="test",
        conversations_db=str(tmp_path / "conversations.db"),
        **overrides,
    )
    return ConversationStore(settings)


def test_conversation_ids_persist_across_turns(tmp_path):
    store = _store(tmp_path)
    first = store.get_or_create(None)
    store.append(first.id, "user", "Who was Einstein?")

    again = store.get_or_create(first.id)

    assert again.id == first.id
    assert [t.content for t in store.history(first.id)] == ["Who was Einstein?"]


def test_history_survives_a_new_store_instance(tmp_path):
    # The point of SQLite: a backend restart must not lose conversations.
    store = _store(tmp_path)
    conversation = store.get_or_create(None)
    store.append(conversation.id, "user", "hello")
    store.close()

    reopened = _store(tmp_path)

    assert [t.content for t in reopened.history(conversation.id)] == ["hello"]


def test_assistant_meta_round_trips(tmp_path):
    store = _store(tmp_path)
    conversation = store.get_or_create(None)
    meta = {"sources": [{"index": 1, "title": "Physics"}]}
    store.append(conversation.id, "assistant", "Fact [1].", meta=meta)

    restored = store.history(conversation.id)[0]

    assert restored.meta == meta


def test_peek_never_creates(tmp_path):
    store = _store(tmp_path)

    assert store.peek("nonexistent") is None

    conversation = store.get_or_create(None, lang="es")
    peeked = store.peek(conversation.id)

    assert peeked is not None
    assert peeked.lang == "es"


def test_history_is_trimmed_to_the_configured_window(tmp_path):
    store = _store(tmp_path, conversation_max_turns=4)
    conversation = store.get_or_create(None)
    for i in range(10):
        store.append(conversation.id, "user", f"message {i}")

    history = store.history(conversation.id)

    assert len(history) == 4
    assert history[-1].content == "message 9"


def test_expired_conversations_are_evicted(tmp_path):
    store = _store(tmp_path, conversation_ttl_seconds=0)
    conversation = store.get_or_create(None)
    store.append(conversation.id, "user", "hello")
    time.sleep(0.01)

    revived = store.get_or_create(conversation.id)

    assert revived.turns == []


def test_clear_removes_a_conversation(tmp_path):
    store = _store(tmp_path)
    conversation = store.get_or_create(None)

    assert store.clear(conversation.id) is True
    assert store.clear(conversation.id) is False
