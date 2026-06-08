"""Streaming STT tests — hermetic: no network, no models, no PortAudio.

We assert the *wiring*: the streaming interface defaults, that the daemon
prefers the streaming path when the backend supports it (feeding live chunks
and storing the note from the session result), and that AudioRecorder delivers
chunks to ``on_chunk``.
"""

from __future__ import annotations

import numpy as np
import pytest

from instant_notes.audio import AudioRecorder
from instant_notes.config import Config
from instant_notes.daemon import InstantNotesDaemon
from instant_notes.stt.base import (
    AudioData,
    StreamingSession,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)


# --- fakes ----------------------------------------------------------------
class FakeStreamingSession(StreamingSession):
    """Records fed chunks; returns a fixed transcript with real metrics."""

    def __init__(self, transcript: str = "hello from the stream") -> None:
        self.transcript = transcript
        self.fed: list[np.ndarray] = []
        self.finished = False
        self._metrics = STTMetrics.start("fake-stream", 0.0)

    def feed(self, chunk: np.ndarray) -> None:
        self.fed.append(np.asarray(chunk, dtype=np.float32))
        self._metrics.mark_first_token()

    def latest_partial(self) -> str:
        return self.transcript

    def finish(self) -> TranscriptionResult:
        self.finished = True
        self._metrics.audio_duration_ms = sum(c.size for c in self.fed) / 16.0
        self._metrics.mark_final()
        return TranscriptionResult(
            text=self.transcript,
            backend="fake-stream",
            metrics=self._metrics,
        )


class FakeStreamingBackend(STTBackend):
    name = "fake-stream"

    def __init__(self, transcript: str = "hello from the stream") -> None:
        self.transcript = transcript
        self.sessions: list[FakeStreamingSession] = []
        self.batch_called = False

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        self.batch_called = True
        return TranscriptionResult(
            text="batch-should-not-be-used",
            backend=self.name,
            metrics=STTMetrics.start(self.name, audio.duration_ms),
        )

    def supports_streaming(self) -> bool:
        return True

    def open_stream(self, sample_rate: int) -> StreamingSession:
        session = FakeStreamingSession(self.transcript)
        self.sessions.append(session)
        return session


class FakeRecorder:
    """Stand-in for AudioRecorder: captures the on_chunk callback and returns
    a fixed AudioData on stop()."""

    sample_rate = 16_000

    def __init__(self) -> None:
        self.on_chunk = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, on_chunk=None) -> None:
        self.on_chunk = on_chunk
        self._recording = True

    def stop(self) -> AudioData:
        self._recording = False
        return AudioData(
            samples=np.zeros(16_000, dtype=np.float32), sample_rate=16_000
        )


# --- interface defaults ---------------------------------------------------
class BatchOnlyBackend(STTBackend):
    name = "batch-only"

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        return TranscriptionResult(
            text="", backend=self.name,
            metrics=STTMetrics.start(self.name, audio.duration_ms),
        )


def test_supports_streaming_default_false():
    backend = BatchOnlyBackend()
    assert backend.supports_streaming() is False


def test_open_stream_raises_on_batch_backend():
    backend = BatchOnlyBackend()
    with pytest.raises(NotImplementedError):
        backend.open_stream(16_000)


# --- AudioRecorder on_chunk ----------------------------------------------
def test_recorder_calls_on_chunk():
    rec = AudioRecorder()
    received: list[np.ndarray] = []
    rec._on_chunk = lambda chunk: received.append(chunk)

    # Mimic a PortAudio block: shape (frames, channels).
    indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    rec._callback(indata, 3, None, None)

    assert len(received) == 1
    # Delivered mono float32, flattened.
    assert received[0].ndim == 1
    np.testing.assert_allclose(received[0], [0.1, 0.2, 0.3], rtol=1e-6)
    # Still buffered for the saved AudioData.
    assert len(rec._frames) == 1


def test_recorder_no_on_chunk_still_buffers():
    rec = AudioRecorder()
    indata = np.array([[0.5]], dtype=np.float32)
    rec._callback(indata, 1, None, None)
    assert len(rec._frames) == 1  # no callback, no crash


# --- daemon prefers streaming --------------------------------------------
def _make_daemon(tmp_path, backend):
    cfg = Config(db_path=tmp_path / "notes.db", notify=False, paste_at_cursor=False)
    daemon = InstantNotesDaemon(cfg)
    daemon.recorder = FakeRecorder()
    # Inject the backend directly so we don't hit the registry.
    daemon._backend = backend
    daemon._backend_error = None
    return daemon


def test_daemon_uses_streaming_path_for_capture(tmp_path):
    backend = FakeStreamingBackend("buy oat milk and call the dentist")
    daemon = _make_daemon(tmp_path, backend)

    # Start recording -> should open a stream and wire on_chunk=session.feed.
    daemon._toggle_capture()
    assert daemon.recorder.is_recording
    assert daemon.recorder.on_chunk is not None
    assert len(backend.sessions) == 1
    session = backend.sessions[0]

    # Simulate two live chunks arriving on the audio thread.
    daemon.recorder.on_chunk(np.array([0.1, 0.2], dtype=np.float32))
    daemon.recorder.on_chunk(np.array([0.3], dtype=np.float32))
    assert len(session.fed) == 2

    # Stop recording -> session.finish() supplies the transcript; note stored.
    daemon._toggle_capture()
    assert session.finished
    assert backend.batch_called is False  # streaming path, never batch

    notes = daemon.store.recent(limit=5)
    assert any("oat milk" in n.text for n in notes)
    daemon.store.close()


def test_daemon_command_uses_streaming(tmp_path):
    backend = FakeStreamingBackend("show notes")
    daemon = _make_daemon(tmp_path, backend)

    daemon._toggle_command()
    assert daemon.recorder.on_chunk is not None
    session = backend.sessions[0]
    daemon.recorder.on_chunk(np.array([0.1], dtype=np.float32))

    daemon._toggle_command()
    assert session.finished
    assert backend.batch_called is False
    daemon.store.close()


def test_daemon_falls_back_to_batch_when_no_streaming(tmp_path):
    backend = BatchOnlyBackend()
    daemon = _make_daemon(tmp_path, backend)

    daemon._toggle_capture()
    # Batch backend: no streaming session, recorder started without on_chunk.
    assert daemon.recorder.is_recording
    assert daemon.recorder.on_chunk is None
    assert daemon._session is None
    daemon._toggle_capture()
    daemon.store.close()


def test_daemon_falls_back_when_open_stream_raises(tmp_path):
    class BrokenStreamingBackend(FakeStreamingBackend):
        def open_stream(self, sample_rate: int) -> StreamingSession:
            raise RuntimeError("socket exploded")

    backend = BrokenStreamingBackend("ignored")
    daemon = _make_daemon(tmp_path, backend)

    daemon._toggle_capture()
    # open_stream raised -> fell back to plain batch recording.
    assert daemon.recorder.is_recording
    assert daemon._session is None
    assert daemon.recorder.on_chunk is None

    daemon._toggle_capture()
    # Batch transcribe() was used for the final transcript.
    assert backend.batch_called is True
    daemon.store.close()
