"""Lightweight latency instrumentation.

A ``Tracer`` records timestamped events for a dictation session and, on
``finalize``, writes one JSON-lines record with derived latencies (notably
``stop_to_text_ms`` — stopped talking → text in front of the user).

Design rules honoured by callers:
- ``mark`` is a non-blocking one-liner at a real boundary.
- marks never raise and never change control flow (all internals are guarded).
- the tracer holds only its own short lock to append a timestamp — callers must
  not call ``mark`` while holding an unrelated lock.

No external dependencies.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_PATH = Path.home() / ".local" / "share" / "instant-notes" / "timing.jsonl"


class Event(StrEnum):
    """Boundaries in a dictation session (START is implicit at ``tracer.start``)."""

    STOP = "stop"                  # recording stopped (key release / VAD silence)
    STT_REQUEST = "stt_request"    # post-stop tail sent to the STT backend
    STT_DONE = "stt_done"          # transcript returned
    DISPLAYED = "displayed"        # raw note persisted + shown (notify/paste begun)
    PASTED = "pasted"              # paste-at-cursor completed
    CLEANUP_START = "cleanup_start"  # background LLM cleanup entered
    CLEANUP_DONE = "cleanup_done"    # note updated with cleaned text


@dataclass
class _Session:
    sid: str
    start: float
    marks: dict[str, float] = field(default_factory=dict)


class Tracer:
    """Collects per-session event timestamps and writes a JSONL record."""

    def __init__(self, path: str | Path | None = None, enabled: bool = True):
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.enabled = enabled
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def start(self) -> str:
        """Begin a session; returns the session id to carry through the flow."""
        sid = uuid.uuid4().hex
        now = time.perf_counter()
        with self._lock:
            self._sessions[sid] = _Session(sid=sid, start=now)
        return sid

    def mark(self, sid: str | None, event: Event) -> None:
        """Record ``event`` for ``sid``. No-op for an unknown/None sid; never raises."""
        if not sid:
            return
        now = time.perf_counter()
        try:
            with self._lock:
                session = self._sessions.get(sid)
                if session is not None:
                    session.marks[Event(event).value] = now
        except Exception:
            pass

    def finalize(self, sid: str | None) -> dict | None:
        """Close ``sid``, write its JSONL record, and return it (or None)."""
        if not sid:
            return None
        try:
            with self._lock:
                session = self._sessions.pop(sid, None)
            if session is None:
                return None
            record = self._build_record(session)
            self._write(record)
            return record
        except Exception:
            return None

    # -- internals ---------------------------------------------------------
    def _build_record(self, session: _Session) -> dict:
        marks = session.marks

        def delta(a: Event, b: Event) -> float | None:
            if a.value in marks and b.value in marks:
                return round(1000.0 * (marks[b.value] - marks[a.value]), 3)
            return None

        # stop → text-in-front-of-user: prefer DISPLAYED, fall back to STT_DONE.
        if Event.DISPLAYED.value in marks:
            stop_to_text = delta(Event.STOP, Event.DISPLAYED)
        else:
            stop_to_text = delta(Event.STOP, Event.STT_DONE)

        events_ms = {
            name: round(1000.0 * (ts - session.start), 3)
            for name, ts in marks.items()
        }
        return {
            "sid": session.sid,
            "events_ms": events_ms,
            "stop_to_text_ms": stop_to_text,
            "stt_ms": delta(Event.STT_REQUEST, Event.STT_DONE),
            "cleanup_ms": delta(Event.CLEANUP_START, Event.CLEANUP_DONE),
            "total_ms": (
                round(1000.0 * (max(marks.values()) - session.start), 3)
                if marks
                else 0.0
            ),
        }

    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


#: process-wide tracer used by the daemon call sites
tracer = Tracer()
