"""The Instant Notes background daemon.

Wires global hotkeys to two flows:

* **capture** — toggle-record a voice note, transcribe it, and store it.
* **command** — toggle-record a spoken query, transcribe it, and route it
  through :class:`~instant_notes.intent.NoteAssistant`.

A single recorder plus a mode flag guards against the two flows colliding: a
new toggle of the *other* kind is ignored while a recording is in flight.
"""

from __future__ import annotations

import time

from instant_notes.audio import AudioRecorder
from instant_notes.config import Config
from instant_notes.hotkeys import HotkeyListener
from instant_notes.intent import NoteAssistant
from instant_notes.notify import notify, paste_at_cursor
from instant_notes.storage import NoteStore
from instant_notes.stt.base import StreamingSession, STTBackend
from instant_notes.stt.registry import build_backend

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
                result = session.finish()
                result.metrics.recording_stopped_ts = stopped_ts
                return audio, result
            except Exception as exc:  # noqa: BLE001 - fall back to batch
                print(f"[instant-notes] streaming finish failed, batch fallback: {exc}")
        result = backend.transcribe(audio)
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
        if transcribed is None:
            return
        audio, result = transcribed
        text = result.text.strip()
        if not text:
            notify("Instant Notes", "Heard nothing.", enabled=self.cfg.notify)
            return
        self.store.add_note(
            text,
            source_backend=result.backend,  # backend name carried on the result
            duration_ms=audio.duration_ms,
            latency_ms=result.metrics.total_ms,
        )
        notify("Note saved", text[:80], enabled=self.cfg.notify)
        if self.cfg.paste_at_cursor:
            paste_at_cursor(text)

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
        if transcribed is None:
            return
        _audio, result = transcribed
        text = result.text.strip()
        if not text:
            notify("Instant Notes", "Heard nothing.", enabled=self.cfg.notify)
            return
        command_result = self.assistant.handle_command(text)
        notify("Notes", command_result.response_text, enabled=self.cfg.notify)
        print(command_result.response_text)

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
            self.store.close()
