"""Core STT contracts. Heavy/optional deps (faster-whisper, groq, deepgram SDKs)
must NOT be imported here — only in the concrete backend modules — so importing
the interface is cheap and never fails on a missing optional package.
"""

from __future__ import annotations

import io
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AudioData:
    """A mono PCM recording. ``samples`` is float32 in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return 1000.0 * len(self.samples) / self.sample_rate

    def to_int16(self) -> np.ndarray:
        clipped = np.clip(self.samples, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

    def to_wav_bytes(self) -> bytes:
        """16-bit PCM WAV — the lowest-common-denominator format every cloud
        STT API accepts."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.to_int16().tobytes())
        return buf.getvalue()


@dataclass
class STTMetrics:
    """Latency instrumentation captured per transcription. Timestamps are
    ``time.perf_counter()`` seconds; only deltas are meaningful."""

    backend: str
    audio_duration_ms: float
    request_sent_ts: float
    final_ts: float
    first_token_ts: Optional[float] = None

    @property
    def total_ms(self) -> float:
        return 1000.0 * (self.final_ts - self.request_sent_ts)

    @property
    def first_token_ms(self) -> Optional[float]:
        if self.first_token_ts is None:
            return None
        return 1000.0 * (self.first_token_ts - self.request_sent_ts)

    @property
    def realtime_factor(self) -> float:
        """total_ms / audio_duration_ms. <1.0 means faster than real time."""
        if self.audio_duration_ms <= 0:
            return 0.0
        return self.total_ms / self.audio_duration_ms

    @classmethod
    def start(cls, backend: str, audio_duration_ms: float) -> "STTMetrics":
        now = time.perf_counter()
        return cls(
            backend=backend,
            audio_duration_ms=audio_duration_ms,
            request_sent_ts=now,
            final_ts=now,
        )

    def mark_first_token(self) -> None:
        if self.first_token_ts is None:
            self.first_token_ts = time.perf_counter()

    def mark_final(self) -> None:
        self.final_ts = time.perf_counter()


@dataclass
class TranscriptionResult:
    text: str
    backend: str
    metrics: STTMetrics
    raw: Optional[dict[str, Any]] = None


class STTBackend(ABC):
    """One pluggable speech-to-text engine. Concrete backends own their SDK and
    config and must populate ``STTMetrics`` honestly so the bench-off is fair."""

    #: stable identifier used in config, CLI, and metrics (e.g. "groq")
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        """Block until the full transcript is available. Streaming backends
        should still return the final text here but set ``first_token_ts`` via
        ``metrics.mark_first_token()`` when the first partial arrives."""

    def is_available(self) -> bool:
        """Whether this backend can run right now (keys present, SDK installed).
        Backends override to check their own preconditions."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<STTBackend {self.name}>"
