"""Local STT backend backed by faster-whisper.

The ``faster_whisper`` package is an *optional* dependency (see
``pyproject.toml`` extras). It is imported lazily — never at module import time —
so this module can be imported (and ``is_available`` queried) even when
faster-whisper is not installed.
"""

from __future__ import annotations

from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)


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
