"""SQLite note storage with full-text search (FTS5).

Schema:
  notes(id, created_at, text, source_backend, duration_ms, latency_ms, audio_path)
  notes_fts  -- FTS5 virtual table mirroring notes.text, kept in sync via triggers
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Note:
    id: int
    created_at: datetime
    text: str
    source_backend: str
    duration_ms: float
    latency_ms: float
    audio_path: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    source_backend TEXT NOT NULL DEFAULT '',
    duration_ms REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    audio_path TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
    USING fts5(text, content='notes', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def _safe_fts_query(query: str) -> str:
    """Turn arbitrary (spoken) text into a safe FTS5 MATCH expression.

    Raw user text can contain FTS operators/quotes (``"``, ``*``, ``:``, ``-``,
    ``NEAR``, unbalanced quotes) that make ``MATCH`` raise and crash command
    handling. We extract word tokens and quote each as a literal term, so the
    query is always valid and operator-free.
    """
    tokens = re.findall(r"\w+", query.lower())
    return " ".join(f'"{t}"' for t in tokens)


def _row_to_note(row: sqlite3.Row) -> Note:
    return Note(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        text=row["text"],
        source_backend=row["source_backend"],
        duration_ms=row["duration_ms"],
        latency_ms=row["latency_ms"],
        audio_path=row["audio_path"],
    )


class NoteStore:
    """Thin DAO over SQLite. Pass ``":memory:"`` for tests."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- writes -----------------------------------------------------------
    def add_note(
        self,
        text: str,
        *,
        source_backend: str = "",
        duration_ms: float = 0.0,
        latency_ms: float = 0.0,
        audio_path: str | None = None,
        created_at: datetime | None = None,
    ) -> Note:
        created_at = created_at or datetime.now(UTC)
        cur = self.conn.execute(
            "INSERT INTO notes (created_at, text, source_backend, duration_ms, "
            "latency_ms, audio_path) VALUES (?, ?, ?, ?, ?, ?)",
            (created_at.isoformat(), text, source_backend, duration_ms,
             latency_ms, audio_path),
        )
        self.conn.commit()
        assert cur.lastrowid is not None  # guaranteed after a successful INSERT
        return Note(
            id=cur.lastrowid,
            created_at=created_at,
            text=text,
            source_backend=source_backend,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            audio_path=audio_path,
        )

    def update_text(self, note_id: int, text: str) -> bool:
        cur = self.conn.execute(
            "UPDATE notes SET text = ? WHERE id = ?", (text, note_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_note(self, note_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # --- reads ------------------------------------------------------------
    def recent(self, limit: int = 20) -> list[Note]:
        rows = self.conn.execute(
            "SELECT * FROM notes ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def get(self, note_id: int) -> Note | None:
        row = self.conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return _row_to_note(row) if row else None

    def since(self, start: datetime) -> list[Note]:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE created_at >= ? ORDER BY created_at ASC",
            (start.isoformat(),),
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def today(self, now: datetime | None = None) -> list[Note]:
        now = now or datetime.now(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.since(start)

    def search(self, query: str, limit: int = 20) -> list[Note]:
        safe = _safe_fts_query(query)
        if not safe:
            return []
        try:
            rows = self.conn.execute(
                "SELECT notes.* FROM notes JOIN notes_fts "
                "ON notes.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? ORDER BY rank LIMIT ?",
                (safe, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Belt-and-suspenders: never let a malformed query crash the caller.
            return []
        return [_row_to_note(r) for r in rows]

    def all(self) -> list[Note]:
        rows = self.conn.execute(
            "SELECT * FROM notes ORDER BY created_at ASC"
        ).fetchall()
        return [_row_to_note(r) for r in rows]
