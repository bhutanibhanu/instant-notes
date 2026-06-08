"""Tests for the voice-command/LLM layer and notification output.

Uses an in-memory NoteStore and a FAKE Claude client — no real API calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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


class FakeClient:
    """Minimal stand-in for anthropic.Anthropic; returns a fixed summary."""

    def __init__(self, text="SUMMARY"):
        self._text = text
        self.calls = []

        client = self

        class _Messages:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(text=client._text)]
                )

        self.messages = _Messages()


@pytest.fixture
def store():
    s = NoteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def cfg():
    c = Config()
    c.answer_model = "claude-opus-4-8"
    c.router_model = "claude-haiku-4-5"
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
    now = datetime.now(timezone.utc)
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


def test_summarize_with_fake_client(store, cfg):
    store.add_note("buy milk")
    fake = FakeClient(text="SUMMARY")
    asst = NoteAssistant(store, cfg, client=fake)

    result = asst.handle_command("summarize my notes")

    assert result.intent == SUMMARIZE
    assert result.response_text == "SUMMARY"
    assert fake.calls  # Claude was actually called
    assert fake.calls[0]["model"] == "claude-opus-4-8"


def test_ask_without_client(store, cfg):
    store.add_note("buy milk")
    asst = NoteAssistant(store, cfg)  # no client, no key

    result = asst.handle_command("how many errands do I have")

    assert result.intent == ASK
    assert "key not configured" in result.response_text.lower()


def test_notify_disabled_prints(capsys):
    rv = notify("t", "m", enabled=False)
    assert rv is None
    captured = capsys.readouterr()
    assert "t: m" in captured.out
