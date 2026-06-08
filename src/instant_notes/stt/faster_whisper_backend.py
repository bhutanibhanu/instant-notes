"""Local STT backend backed by faster-whisper.

The ``faster_whisper`` package is an *optional* dependency (see
``pyproject.toml`` extras). It is imported lazily — never at module import time —
so this module can be imported (and ``is_available`` queried) even when
faster-whisper is not installed.

Streaming here is *windowed*: there is no incremental Whisper decoder, so the
session re-transcribes the whole accumulated buffer every few seconds to
produce a running partial. See ``docs/adr/0001-streaming-vs-batch.md`` for the
O(n^2) tradeoff (acceptable for short voice notes).
"""

from __future__ import annotations

import threading

import numpy as np

from instant_notes.stt.base import (
    AudioData,
    StreamingSession,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)

#: Re-transcribe once this much *new* audio has accumulated (seconds).
PARTIAL_INTERVAL_S = 3.0


class FasterWhisperBackend(STTBackend):
    """Run Whisper locally on CPU via faster-whisper (CTranslate2)."""

    name = "faster-whisper"

    def __init__(self, model: str = "small", compute_type: str = "int8") -> None:
        self.model = model
        self.compute_type = compute_type
        self._model = None  # lazily loaded WhisperModel, cached on the instance

    def _ensure_model(self):
        """Load and cache the WhisperModel on first use."""
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import

            # device="cpu": CTranslate2 has no Apple-Metal backend, so int8 on
            # CPU is the correct (and fastest available) choice on macOS.
            self._model = WhisperModel(
                self.model, device="cpu", compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        model = self._ensure_model()

        metrics = STTMetrics.start(self.name, audio.duration_ms)

        # faster-whisper accepts a float32 numpy array directly. beam_size=1
        # keeps it fast (greedy decoding). segments is a lazy generator — work
        # only happens as we iterate it.
        segments, info = model.transcribe(audio.samples, beam_size=1)

        parts: list[str] = []
        for i, segment in enumerate(segments):
            if i == 0:
                metrics.mark_first_token()
            parts.append(segment.text)
        metrics.mark_final()

        text = "".join(parts).strip()
        language = getattr(info, "language", None)
        return TranscriptionResult(
            text=text,
            backend=self.name,
            metrics=metrics,
            raw={"language": language},
        )

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def supports_streaming(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def open_stream(self, sample_rate: int) -> StreamingSession:
        model = self._ensure_model()
        return _FasterWhisperStreamingSession(model, sample_rate, self.name)


class _FasterWhisperStreamingSession(StreamingSession):
    """Windowed local streaming session.

    Fed chunks are appended to an in-memory buffer. A worker thread wakes up
    whenever ~``PARTIAL_INTERVAL_S`` of new audio has arrived and re-transcribes
    the WHOLE buffer to refresh the partial. :meth:`finish` runs one final pass
    over the complete buffer.
    """

    def __init__(self, model, sample_rate: int, backend_name: str) -> None:
        self._model = model
        self._sample_rate = sample_rate or 16_000
        self._backend_name = backend_name
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._partial = ""
        self._closed = False
        self._fed_samples = 0
        self._last_partial_samples = 0
        self._wake = threading.Event()
        self._interval_samples = int(PARTIAL_INTERVAL_S * self._sample_rate)

        self._metrics = STTMetrics.start(backend_name, 0.0)
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def feed(self, chunk: np.ndarray) -> None:
        if self._closed:
            return
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            self._chunks.append(arr)
            self._fed_samples += arr.size
            due = (
                self._fed_samples - self._last_partial_samples
                >= self._interval_samples
            )
        if due:
            self._wake.set()

    def _snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks, axis=0)

    def _transcribe_buffer(self, buffer: np.ndarray) -> str:
        if buffer.size == 0:
            return ""
        segments, _info = self._model.transcribe(buffer, beam_size=1)
        return "".join(seg.text for seg in segments).strip()

    def _loop(self) -> None:
        while not self._closed:
            # Wait until enough new audio accrues (or a periodic timeout so we
            # don't hang if chunks are tiny).
            self._wake.wait(timeout=PARTIAL_INTERVAL_S)
            self._wake.clear()
            if self._closed:
                return
            with self._lock:
                self._last_partial_samples = self._fed_samples
            buffer = self._snapshot()
            text = self._transcribe_buffer(buffer)
            if text:
                self._metrics.mark_first_token()
                with self._lock:
                    self._partial = text

    def latest_partial(self) -> str:
        with self._lock:
            return self._partial

    def finish(self) -> TranscriptionResult:
        self._closed = True
        self._wake.set()
        self._worker.join(timeout=1.0)

        self._metrics.audio_duration_ms = (
            1000.0 * self._fed_samples / self._sample_rate
            if self._sample_rate
            else 0.0
        )
        try:
            buffer = self._snapshot()
            text = self._transcribe_buffer(buffer)
            if text:
                self._metrics.mark_first_token()
        except Exception:
            with self._lock:
                text = self._partial
        self._metrics.mark_final()
        return TranscriptionResult(
            text=text,
            backend=self._backend_name,
            metrics=self._metrics,
            raw={"streaming": True},
        )
