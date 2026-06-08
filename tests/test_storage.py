from datetime import UTC, datetime, timedelta

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
    now = datetime.now(UTC)
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


def test_update_text():
    store = NoteStore(":memory:")
    n = store.add_note("aaj ma eek note")
    assert store.update_text(n.id, "aaj main ek note") is True
    assert store.get(n.id).text == "aaj main ek note"
    # FTS reflects the update
    assert len(store.search("main")) == 1
    assert store.update_text(9999, "nope") is False


def test_search_with_fts_special_chars_does_not_crash():
    """Spoken queries with FTS operators/quotes/punctuation must never raise."""
    store = NoteStore(":memory:")
    store.add_note("email me at user@example.com about the milk")
    store.add_note("buy milk - eggs")
    # These would all raise sqlite3.OperationalError if passed raw to MATCH.
    for q in [
        "user@example.com",
        "milk - eggs",
        'unbalanced "quote',
        "NEAR/3 foo",
        "*",
        "AND OR NOT",
        "",
        "   ",
    ]:
        result = store.search(q)  # must not raise
        assert isinstance(result, list)


def test_search_special_chars_still_finds_terms():
    store = NoteStore(":memory:")
    store.add_note("email me at user@example.com about milk")
    # Tokenized to literal terms — still matches the note containing them.
    hits = store.search("user@example.com")
    assert len(hits) == 1
