from datetime import datetime, timedelta, timezone

from instant_notes.storage import NoteStore


def test_add_and_recent():
    store = NoteStore(":memory:")
    store.add_note("first note", source_backend="groq")
    n2 = store.add_note("second note", source_backend="deepgram")
    recent = store.recent()
    assert len(recent) == 2
    assert recent[0].id == n2.id  # newest first
    assert recent[0].source_backend == "deepgram"


def test_search_fts():
    store = NoteStore(":memory:")
    store.add_note("buy milk and eggs")
    store.add_note("call the dentist")
    hits = store.search("milk")
    assert len(hits) == 1
    assert "milk" in hits[0].text


def test_today_filter():
    store = NoteStore(":memory:")
    now = datetime.now(timezone.utc)
    store.add_note("old", created_at=now - timedelta(days=2))
    store.add_note("fresh", created_at=now)
    today = store.today(now=now)
    assert [n.text for n in today] == ["fresh"]


def test_delete():
    store = NoteStore(":memory:")
    n = store.add_note("temp")
    assert store.delete_note(n.id) is True
    assert store.get(n.id) is None
    assert store.search("temp") == []
