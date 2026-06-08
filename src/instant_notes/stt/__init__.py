"""Speech-to-text backends. All implement ``STTBackend`` from ``base``."""

from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)

__all__ = ["AudioData", "STTBackend", "STTMetrics", "TranscriptionResult"]
