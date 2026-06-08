"""The Instant Notes background daemon.

Wires global hotkeys to two flows:

* **capture** — toggle-record a voice note, transcribe it, and store it.
* **command** — toggle-record a spoken query, transcribe it, and route it
  through :class:`~instant_notes.intent.NoteAssistant`.

A single recorder plus a mode flag guards against the two flows colliding: a
new toggle of the *other* kind is ignored while a recording is in flight.
"""

from __future__ import annotations

import threading
import time

from instant_notes import llm
from instant_notes.audio import AudioRecorder, trim_silence
from instant_notes.config import Config
from instant_notes.hotkeys import HotkeyListener
from instant_notes.intent import NoteAssistant
from instant_notes.notify import notify, paste_at_cursor
from instant_notes.refine import clean_transcript
from instant_notes.storage import NoteStore
from instant_notes.stt.base import StreamingSession, STTBackend
from instant_notes.stt.registry import build_backend
from instant_notes.timing import Event, tracer

# recording-mode flags
_IDLE = None
_CAPTURE = "capture"
_COMMAND = "command"


class InstantNotesDaemon:
    """Long-running process that owns the recorder, store, STT backend and the
    hotkey listener."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = NoteStore(cfg.db_path)
        self.assistant = NoteAssistant(self.store, cfg)
        self.recorder = AudioRecorder()
        self._mode: str | None = _IDLE
        # STT backend is built lazily so a missing optional dep doesn't stop
        # the daemon from loading.
        self._backend: STTBackend | None = None
        self._backend_error: str | None = None
        # Live streaming session for the in-flight recording, if the backend
        # supports streaming. None means we're on the batch path.
        self._session: StreamingSession | None = None
        # Latency-trace id for the in-flight dictation (carried on the daemon,
        # not a global). Set at record start, finalized after cleanup.
        self._sid: str | None = None

    # --- backend ----------------------------------------------------------
    @property
    def backend(self) -> STTBackend | None:
        """The capture STT backend, constructed on first use."""
        if self._backend is None and self._backend_error is None:
            try:
                self._backend = build_backend(self.cfg.default_backend, self.cfg)
            except Exception as exc:  # missing dep / key / model
                self._backend_error = str(exc)
                print(
                    f"[instant-notes] could not load STT backend "
                    f"{self.cfg.default_backend!r}: {exc}"
                )
        return self._backend

    def _start_recording(self) -> None:
        """Begin capturing. If the backend can stream, open a live session and
        feed audio to it during recording; otherwise plain batch recording.

        On any failure opening the stream we log and fall back to batch."""
        self._sid = tracer.start()
        self._session = None
        backend = self.backend
        if backend is not None and backend.supports_streaming():
            try:
                session = backend.open_stream(self.recorder.sample_rate)
                self.recorder.start(on_chunk=session.feed)
                self._session = session
                return
            except Exception as exc:  # noqa: BLE001 - fall back to batch
                print(f"[instant-notes] streaming unavailable, using batch: {exc}")
                self._session = None
        self.recorder.start()

    def _transcribe(self):
        """Stop recording and transcribe. Returns (audio, result) or None if no
        backend is available.

        When a live streaming session is in flight we still call
        ``recorder.stop()`` (to keep the full saved AudioData + duration) but
        take the transcript from the session's ``finish()`` for low latency."""
        session = self._session
        self._session = None
        audio = self.recorder.stop()
        stopped_ts = time.perf_counter()  # the moment the user stopped talking
        tracer.mark(self._sid, Event.STOP)
        backend = self.backend
        if backend is None:
            if session is not None:
                try:
                    session.finish()
                except Exception:
                    pass
            notify(
                "Instant Notes",
                "No STT backend available — check config/keys.",
                enabled=self.cfg.notify,
            )
            return None
        if session is not None:
            try:
                tracer.mark(self._sid, Event.STT_REQUEST)
                result = session.finish()
                tracer.mark(self._sid, Event.STT_DONE)
                result.metrics.recording_stopped_ts = stopped_ts
                return audio, result
            except Exception as exc:  # noqa: BLE001 - fall back to batch
                print(f"[instant-notes] streaming finish failed, batch fallback: {exc}")
        # Batch path: trim dead air before sending (fewer bytes, less hallucination).
        clip = trim_silence(audio) if self.cfg.trim_silence else audio
        tracer.mark(self._sid, Event.STT_REQUEST)
        result = backend.transcribe(clip)
        tracer.mark(self._sid, Event.STT_DONE)
        result.metrics.recording_stopped_ts = stopped_ts
        return audio, result

    # --- capture ----------------------------------------------------------
    def _toggle_capture(self) -> None:
        if self._mode == _COMMAND:
            return  # a command recording is in flight; ignore
        if not self.recorder.is_recording:
            self._start_recording()
            self._mode = _CAPTURE
            notify("Instant Notes", "Recording…", enabled=self.cfg.notify)
            return

        self._mode = _IDLE
        transcribed = self._transcribe()
        sid, self._sid = self._sid, None  # take ownership of the trace id
        if transcribed is None:
            tracer.finalize(sid)
            return
        audio, result = transcribed
        text = result.text.strip()
        if not text:
            notify("Instant Notes", "Heard nothing.", enabled=self.cfg.notify)
            tracer.finalize(sid)
            return
        # Save + surface the raw transcript immediately (instant feel), then
        # refine it in the background and update the note when cleanup returns.
        note = self.store.add_note(
            text,
            source_backend=result.backend,  # backend name carried on the result
            duration_ms=audio.duration_ms,
            latency_ms=result.metrics.total_ms,
        )
        notify("Note saved", text[:80], enabled=self.cfg.notify)
        tracer.mark(sid, Event.DISPLAYED)
        if self.cfg.paste_at_cursor:
            paste_at_cursor(text)
            tracer.mark(sid, Event.PASTED)
        if self.cfg.llm_cleanup and llm.available(self.cfg):
            threading.Thread(
                target=self._refine_note, args=(note.id, text, sid), daemon=True
            ).start()
        else:
            tracer.finalize(sid)

    def _refine_note(self, note_id: int, raw: str, sid: str | None = None) -> None:
        """Background: LLM-clean the transcript and update the saved note. Uses
        its own DB connection (SQLite connections aren't shared across threads)."""
        tracer.mark(sid, Event.CLEANUP_START)
        cleaned = clean_transcript(raw, self.cfg)
        if cleaned and cleaned != raw:
            store = NoteStore(self.cfg.db_path)
            try:
                store.update_text(note_id, cleaned)
            finally:
                store.close()
            notify("Note refined", cleaned[:80], enabled=self.cfg.notify)
        tracer.mark(sid, Event.CLEANUP_DONE)
        tracer.finalize(sid)

    # --- command ----------------------------------------------------------
    def _toggle_command(self) -> None:
        if self._mode == _CAPTURE:
            return  # a capture recording is in flight; ignore
        if not self.recorder.is_recording:
            self._start_recording()
            self._mode = _COMMAND
            notify("Instant Notes", "Listening…", enabled=self.cfg.notify)
            return

        self._mode = _IDLE
        transcribed = self._transcribe()
        sid, self._sid = self._sid, None
        if transcribed is None:
            tracer.finalize(sid)
            return
        _audio, result = transcribed
        text = result.text.strip()
        if not text:
            notify("Instant Notes", "Heard nothing.", enabled=self.cfg.notify)
            tracer.finalize(sid)
            return
        command_result = self.assistant.handle_command(text)
        notify("Notes", command_result.response_text, enabled=self.cfg.notify)
        print(command_result.response_text)
        tracer.finalize(sid)

    # --- lifecycle --------------------------------------------------------
    def run(self) -> None:
        listener = HotkeyListener(
            {
                self.cfg.capture_hotkey: self._toggle_capture,
                self.cfg.command_hotkey: self._toggle_command,
            }
        )
        listener.start()
        print("Instant Notes daemon ready.")
        print(f"  capture:  {self.cfg.capture_hotkey}")
        print(f"  command:  {self.cfg.command_hotkey}")
        print(f"  backend:  {self.cfg.default_backend}")
        print("Press Ctrl-C to quit.")
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nShutting down…")
        finally:
            listener.stop()
            if self.recorder.is_recording:
                self.recorder.stop()
            # Clean up an in-flight streaming session (thread pool + sockets).
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = None
            # Close any backend that holds connections (pooled httpx, etc.).
            close = getattr(self._backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self.store.close()
