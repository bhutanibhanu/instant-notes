"""Deepgram speech-to-text backend (nova-3).

Uses the ``deepgram-sdk`` v3 client, imported lazily so this module imports
cleanly even when the package is not installed.
"""

from __future__ import annotations

from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)


class DeepgramBackend(STTBackend):
    """Deepgram cloud STT (pre-recorded transcription)."""

    name = "deepgram"

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self.api_key = api_key
        self.model = model

    def is_available(self) -> bool:
        try:
            import deepgram  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        from deepgram import DeepgramClient, PrerecordedOptions

        wav_bytes = audio.to_wav_bytes()
        dg = DeepgramClient(self.api_key)
        options = PrerecordedOptions(model=self.model, smart_format=True)

        metrics = STTMetrics.start(self.name, audio.duration_ms)
        resp = dg.listen.rest.v("1").transcribe_file({"buffer": wav_bytes}, options)
        metrics.mark_final()

        text = self._extract_transcript(resp).strip()
        raw = self._to_raw(resp)

        return TranscriptionResult(
            text=text,
            backend=self.name,
            metrics=metrics,
            raw=raw,
        )

    @staticmethod
    def _extract_transcript(resp: object) -> str:
        try:
            return resp.results.channels[0].alternatives[0].transcript or ""
        except (AttributeError, IndexError, TypeError):
            pass
        # Fall back to dict-shaped responses.
        try:
            return (
                resp["results"]["channels"][0]["alternatives"][0]["transcript"] or ""
            )
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _to_raw(resp: object) -> dict:
        for attr in ("to_dict", "model_dump", "dict"):
            fn = getattr(resp, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        if isinstance(resp, dict):
            return resp
        return {"transcript": DeepgramBackend._extract_transcript(resp)}
