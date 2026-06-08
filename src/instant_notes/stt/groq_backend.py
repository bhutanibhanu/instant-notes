"""Groq speech-to-text backend (whisper-large-v3-turbo).

The ``groq`` SDK is imported lazily so this module imports cleanly even when the
package is not installed.
"""

from __future__ import annotations

from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)


class GroqBackend(STTBackend):
    """Groq-hosted Whisper STT."""

    name = "groq"

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None  # cached Groq client (connection reuse)

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

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        wav_bytes = audio.to_wav_bytes()
        client = self._groq()

        metrics = STTMetrics.start(self.name, audio.duration_ms)
        resp = client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=self.model,
            response_format="json",
        )
        metrics.mark_final()

        text = (getattr(resp, "text", "") or "").strip()
        raw = self._to_raw(resp)

        return TranscriptionResult(
            text=text,
            backend=self.name,
            metrics=metrics,
            raw=raw,
        )

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
