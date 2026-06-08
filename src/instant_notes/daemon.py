"""The Instant Notes background daemon.

Wires global hotkeys to two flows:

* **capture** — toggle-record a voice note, transcribe it, and store it.
* **command** — toggle-record a spoken query, transcribe it, and route it
  through :class:`~instant_notes.intent.NoteAssistant`.

A single recorder plus a mode flag guards against the two flows colliding: a
new toggle of the *other* kind is ignored while a recording is in flight.
"""

from __future__ import annotations

from typing import Optional

from instant_notes.audio import AudioRecorder
from instant_notes.config import Config
from instant_notes.hotkeys import HotkeyListener
from instant_notes.intent import NoteAssistant
from instant_notes.notify import notify, paste_at_cursor
from instant_notes.storage import NoteStore
from instant_notes.stt.base import STTBackend
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
        self._mode: Optional[str] = _IDLE
        # STT backend is built lazily so a missing optional dep doesn't stop
        # the daemon from loading.
        self._backend: Optional[STTBackend] = None
        self._backend_error: Optional[str] = None

    # --- backend ----------------------------------------------------------
    @property
    def backend(self) -> Optional[STTBackend]:
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

    def _transcribe(self):
        """Stop recording and transcribe. Returns (audio, result) or None if no
        backend is available."""
        audio = self.recorder.stop()
        backend = self.backend
        if backend is None:
            notify(
                "Instant Notes",
                "No STT backend available — check config/keys.",
                enabled=self.cfg.notify,
            )
            return None
        result = backend.transcribe(audio)
        return audio, result

    # --- capture ----------------------------------------------------------
    def _toggle_capture(self) -> None:
        if self._mode == _COMMAND:
            return  # a command recording is in flight; ignore
        if not self.recorder.is_recording:
            self.recorder.start()
            self._mode = _CAPTURE
            notify("Instant Notes", "Recording…", enabled=self.cfg.notify)
            return

        self._mode = _IDLE
        transcribed = self._transcribe()
        if transcribed is None:
            return
        audio, result = transcribed
        backend = self.backend
        text = result.text.strip()
        if not text:
            notify("Instant Notes", "Heard nothing.", enabled=self.cfg.notify)
            return
        self.store.add_note(
            text,
            source_backend=backend.name,
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
            self.recorder.start()
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
