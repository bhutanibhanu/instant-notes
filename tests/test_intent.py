"""Tests for the voice-command/LLM layer and notification output.

Uses an in-memory NoteStore and a FAKE chat_fn — no real API calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from instant_notes.config import Config
from instant_notes.intent import (
    ASK,
    SEARCH,
    SHOW_RECENT,
    SUMMARIZE,
    TODAY,
    NoteAssistant,
)
from instant_notes.notify import notify
from instant_notes.storage import NoteStore


@pytest.fixture
def store():
    s = NoteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def cfg():
    c = Config()
    c.groq_api_key = None       # no LLM unless a chat_fn is injected
    c.anthropic_api_key = None
    return c


def test_show_recent(store, cfg):
    store.add_note("buy milk")
    store.add_note("call mom")
    asst = NoteAssistant(store, cfg)

    result = asst.handle_command("show my notes")

    assert result.intent == SHOW_RECENT
    assert "buy milk" in result.response_text
    assert "call mom" in result.response_text
    assert len(result.notes) == 2


def test_show_recent_empty(store, cfg):
    asst = NoteAssistant(store, cfg)
    result = asst.handle_command("list my notes")
    assert result.intent == SHOW_RECENT
    assert result.response_text == "No notes yet."


def test_today(store, cfg):
    now = datetime.now(UTC)
    yesterday = now - timedelta(days=1)
    store.add_note("today note", created_at=now)
    store.add_note("old note", created_at=yesterday)

    asst = NoteAssistant(store, cfg)
    result = asst.handle_command("what did I note today")

    assert result.intent == TODAY
    assert "today note" in result.response_text
    assert "old note" not in result.response_text
    assert len(result.notes) == 1


def test_search(store, cfg):
    store.add_note("buy milk")
    store.add_note("call mom")
    asst = NoteAssistant(store, cfg)

    result = asst.handle_command("search for milk")

    assert result.intent == SEARCH
    assert "buy milk" in result.response_text
    assert any("milk" in n.text for n in result.notes)


def test_summarize_with_fake_chat(store, cfg):
    store.add_note("buy milk")
    calls = []

    def chat_fn(system, user):
        calls.append((system, user))
        return "SUMMARY"

    asst = NoteAssistant(store, cfg, chat_fn=chat_fn)
    result = asst.handle_command("summarize my notes")

    assert result.intent == SUMMARIZE
    assert result.response_text == "SUMMARY"
    assert calls  # the LLM was actually called
    assert "buy milk" in calls[0][1]  # notes passed to the model


def test_ask_without_llm(store, cfg):
    store.add_note("buy milk")
    asst = NoteAssistant(store, cfg)  # no chat_fn, no keys

    result = asst.handle_command("how many errands do I have")

    assert result.intent == ASK
    assert "no llm configured" in result.response_text.lower()


def test_ask_with_fake_chat(store, cfg):
    store.add_note("buy milk")
    asst = NoteAssistant(store, cfg, chat_fn=lambda s, u: "You have 1 errand.")
    result = asst.handle_command("how many errands do I have")
    assert result.intent == ASK
    assert result.response_text == "You have 1 errand."


def test_notify_disabled_prints(capsys):
    rv = notify("t", "m", enabled=False)
    assert rv is None
    captured = capsys.readouterr()
    assert "t: m" in captured.out
