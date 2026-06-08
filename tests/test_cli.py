"""CLI tests — hermetic: no audio, hotkeys, or network.

The daemon/bench code paths that touch hardware or remote APIs are not
exercised here; we cover ``version`` and ``notes`` (which only need SQLite) plus
``bench`` argument validation.
"""

from __future__ import annotations

import pytest

from instant_notes import __version__
from instant_notes.cli import main
from instant_notes.config import Config
from instant_notes.storage import NoteStore


def test_version(capsys):
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_notes_lists_seeded_note(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "notes.db"

    # Seed a note in the temp db.
    seed = NoteStore(db_path)
    seed.add_note(
        "remember to buy oat milk",
        source_backend="groq",
        duration_ms=1200.0,
        latency_ms=300.0,
    )
    seed.close()

    cfg = Config(db_path=db_path)
    monkeypatch.setattr("instant_notes.cli.load_config", lambda path=None: cfg)

    rc = main(["notes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "oat milk" in out
    assert "groq" in out


def test_notes_empty(tmp_path, monkeypatch, capsys):
    cfg = Config(db_path=tmp_path / "empty.db")
    monkeypatch.setattr("instant_notes.cli.load_config", lambda path=None: cfg)

    rc = main(["notes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No notes found" in out


def test_notes_search(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "notes.db"
    seed = NoteStore(db_path)
    seed.add_note("buy oat milk", source_backend="groq")
    seed.add_note("call the dentist", source_backend="groq")
    seed.close()

    cfg = Config(db_path=db_path)
    monkeypatch.setattr("instant_notes.cli.load_config", lambda path=None: cfg)

    rc = main(["notes", "--search", "dentist"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dentist" in out
    assert "oat milk" not in out


def test_bench_requires_audio():
    # argparse should exit because --audio is required.
    with pytest.raises(SystemExit) as exc:
        main(["bench"])
    assert exc.value.code != 0


def test_unknown_command_errors():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
