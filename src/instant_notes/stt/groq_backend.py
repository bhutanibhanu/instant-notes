"""Groq speech-to-text backend (whisper-large-v3-turbo).

The ``groq`` SDK is imported lazily so this module imports cleanly even when the
package is not installed.

Two output modes (``output_script``):
- ``"native"``  — Whisper default: Hindi in Devanagari, English in Latin.
- ``"roman"``   — romanized Hinglish (Hindi in Latin letters). Achieved with
  ``language="en"`` + a romanized-Hinglish ``prompt``; long audio is windowed and
  the prompt re-primed per window to keep romanization from drifting.

Streaming (``open_stream``): Groq is a batch API, but we approximate live
transcription by transcribing fixed windows *as audio is fed during recording*.
When the user stops, only the final partial window remains — so stop→text latency
is roughly one window, not the whole clip.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from instant_notes.stt.base import (
    AudioData,
    StreamingSession,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)

DEFAULT_ROMANIZE_PROMPT = (
    "Namaste, mera naam Bhanu hai. Yeh ek Hinglish transcription hai jisme Hindi "
    "words roman letters mein likhe gaye hain aur English words normal hain. "
    "Maine socha ki aaj kuch test karu."
)
ROMANIZE_CHUNK_S = 20.0   # window for re-priming the prompt (batch romanized)
# Streaming windows are flushed at a natural pause once >= MIN, or forced at MAX
# (hard cap to bound latency). Pause-aligned cuts avoid slicing mid-word, which
# hurts accuracy at window boundaries.
STREAM_MIN_WINDOW_S = 7.0
STREAM_MAX_WINDOW_S = 14.0
STREAM_TAIL_MS = 300.0      # trailing audio inspected for a pause
STREAM_SILENCE_RMS = 0.015  # below this = silence


class GroqBackend(STTBackend):
    """Groq-hosted Whisper STT."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        output_script: str = "native",
        romanize_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.output_script = output_script
        self.romanize_prompt = romanize_prompt or DEFAULT_ROMANIZE_PROMPT
        self._client: Any = None  # cached Groq client (connection reuse)

    def is_available(self) -> bool:
        try:
            import groq  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def _groq(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def _transcribe_window(self, audio: AudioData) -> str:
        """One Groq transcription call for a single window (honours output_script)."""
        kwargs: dict[str, Any] = {}
        if self.output_script == "roman":
            kwargs = {"language": "en", "prompt": self.romanize_prompt}
        resp = self._groq().audio.transcriptions.create(
            file=("audio.wav", audio.to_wav_bytes()),
            model=self.model,
            response_format="json",
            **kwargs,
        )
        return (getattr(resp, "text", "") or "").strip()

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        chunks = self._chunks(audio) if self.output_script == "roman" else [audio]
        metrics = STTMetrics.start(self.name, audio.duration_ms)
        if len(chunks) == 1:
            parts = [self._transcribe_window(chunks[0])]
        else:
            # Independent windows (fixed prompt) → run concurrently.
            with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
                parts = list(pool.map(self._transcribe_window, chunks))
        metrics.mark_first_token()
        metrics.mark_final()
        return TranscriptionResult(
            text=" ".join(p for p in parts if p).strip(),
            backend=self.name,
            metrics=metrics,
            raw={"windows": len(chunks)},
        )

    # --- streaming (transcribe-while-recording) ---------------------------
    def supports_streaming(self) -> bool:
        return self.is_available()

    def open_stream(self, sample_rate: int) -> StreamingSession:
        return _GroqStreamingSession(self, sample_rate)

    @staticmethod
    def _chunks(audio: AudioData) -> list[AudioData]:
        win = int(ROMANIZE_CHUNK_S * audio.sample_rate)
        n = len(audio.samples)
        if n <= win:
            return [audio]
        return [
            AudioData(samples=audio.samples[i : i + win], sample_rate=audio.sample_rate)
            for i in range(0, n, win)
        ]

    @staticmethod
    def _to_raw(resp: object) -> dict:
        for attr in ("model_dump", "to_dict", "dict"):
            fn = getattr(resp, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return {"text": getattr(resp, "text", None)}


class _GroqStreamingSession(StreamingSession):
    """Transcribe ~GROQ_STREAM_WINDOW_S windows as audio is fed during recording.

    ``feed`` is non-blocking: it buffers and, once a full window accrues, submits
    that window to a thread pool. ``finish`` flushes the tail and joins, so the
    only work left at stop is the final partial window.
    """

    def __init__(self, backend: GroqBackend, sample_rate: int) -> None:
        self._b = backend
        self._sr = sample_rate
        self._min = int(STREAM_MIN_WINDOW_S * sample_rate)
        self._max = int(STREAM_MAX_WINDOW_S * sample_rate)
        self._tail = int(STREAM_TAIL_MS / 1000.0 * sample_rate)
        self._buf: list[np.ndarray] = []
        self._buf_n = 0
        self._fed = 0
        self._futures: list[Any] = []
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        self._metrics = STTMetrics.start(backend.name, 0.0)

    def feed(self, chunk: np.ndarray) -> None:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            self._fed += arr.size
            self._buf.append(arr)
            self._buf_n += arr.size
            # Hard cap: never let a window grow past MAX (bounds latency).
            while self._buf_n >= self._max:
                self._submit(self._pop_locked(self._max))
            # Pause-aligned flush: once we have enough audio and the tail is
            # silent, cut here so windows end on phrase boundaries.
            if self._buf_n >= self._min and self._trailing_silence_locked():
                self._submit(self._pop_locked(self._buf_n))

    def _trailing_silence_locked(self) -> bool:
        data = self._buf[-1] if len(self._buf) == 1 else np.concatenate(self._buf)
        tail = data[-self._tail :]
        if tail.size == 0:
            return False
        rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        return rms < STREAM_SILENCE_RMS

    def _pop_locked(self, n: int) -> np.ndarray:
        data = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        window, rest = data[:n], data[n:]
        self._buf = [rest] if rest.size else []
        self._buf_n = int(rest.size)
        return window

    def _submit(self, samples: np.ndarray) -> None:
        audio = AudioData(samples=samples, sample_rate=self._sr)
        self._futures.append(self._pool.submit(self._b._transcribe_window, audio))

    def finish(self) -> TranscriptionResult:
        with self._lock:
            if self._buf_n > 0:
                self._submit(self._pop_locked(self._buf_n))
        parts: list[str] = []
        for i, fut in enumerate(self._futures):
            try:
                t = fut.result()
            except Exception:
                t = ""
            if i == 0:
                self._metrics.mark_first_token()
            if t:
                parts.append(t)
        self._metrics.audio_duration_ms = (
            1000.0 * self._fed / self._sr if self._sr else 0.0
        )
        self._metrics.mark_final()
        self._pool.shutdown(wait=False)
        return TranscriptionResult(
            text=" ".join(parts).strip(),
            backend=self._b.name,
            metrics=self._metrics,
            raw={"windows": len(self._futures)},
        )
