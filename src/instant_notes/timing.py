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
            # perceived "instant feel": stopped talking → text in front of user
            "stop_to_text_ms": stop_to_text,
            # post-stop STT of the live tail (request→done). In streaming mode this
            # is the only audio not already transcribed during recording.
            "tail_stt_ms": delta(Event.STT_REQUEST, Event.STT_DONE),
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


# --- stats (CLI: `python -m instant_notes.timing stats`) -------------------

STAT_METRICS = ["stop_to_text_ms", "tail_stt_ms", "cleanup_ms", "total_ms"]


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pure stdlib)."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_records(path: str | Path) -> list[dict]:
    """Read JSONL timing records; tolerate a missing file / bad lines."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def compute_stats(records: list[dict]) -> dict[str, dict[str, float]]:
    """Per-metric {n, p50, p95, max} over the records (None values skipped)."""
    out: dict[str, dict[str, float]] = {}
    for metric in STAT_METRICS:
        vals = [
            float(r[metric])
            for r in records
            if isinstance(r.get(metric), (int, float))
        ]
        if vals:
            out[metric] = {
                "n": len(vals),
                "p50": _percentile(vals, 50),
                "p95": _percentile(vals, 95),
                "max": max(vals),
            }
    return out


def _print_stats(path: str | Path) -> int:
    records = load_records(path)
    if not records:
        print(f"No timing data at {path}.")
        print("Run `instant-notes start` and dictate a few notes first.")
        return 0
    stats = compute_stats(records)
    print(f"{len(records)} sessions  ({path})\n")
    print(f"{'metric':<18}{'n':>5}{'p50':>10}{'p95':>10}{'max':>10}")
    print("-" * 53)
    for metric in STAT_METRICS:
        s = stats.get(metric)
        if not s:
            continue
        print(
            f"{metric:<18}{int(s['n']):>5}"
            f"{s['p50']:>10.1f}{s['p95']:>10.1f}{s['max']:>10.1f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m instant_notes.timing")
    sub = parser.add_subparsers(dest="cmd")
    p_stats = sub.add_parser("stats", help="p50/p95/max per latency metric")
    p_stats.add_argument("--path", default=str(DEFAULT_PATH), help="JSONL log path")
    args = parser.parse_args(argv)
    if args.cmd == "stats":
        return _print_stats(args.path)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
