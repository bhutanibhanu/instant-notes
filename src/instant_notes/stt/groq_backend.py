"""Groq speech-to-text backend (whisper-large-v3-turbo).

The ``groq`` SDK is imported lazily so this module imports cleanly even when the
package is not installed.

Two output modes (``output_script``):
- ``"native"``  — Whisper's default: Hindi in Devanagari, English in Latin.
- ``"roman"``   — romanized Hinglish (Hindi written in Latin letters), which is
  how the user actually types notes. Achieved with ``language="en"`` + a
  romanized-Hinglish ``prompt`` that biases Whisper's output script. Whisper
  drifts back to Devanagari over long audio, so in roman mode we transcribe in
  short windows, re-priming the prompt each chunk to keep romanization stable.
"""

from __future__ import annotations

from typing import Any

from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)

# A romanized-Hinglish exemplar; priming Whisper with this biases it to emit
# Hindi in Latin letters while keeping English words intact.
DEFAULT_ROMANIZE_PROMPT = (
    "Namaste, mera naam Bhanu hai. Yeh ek Hinglish transcription hai jisme Hindi "
    "words roman letters mein likhe gaye hain aur English words normal hain. "
    "Maine socha ki aaj kuch test karu."
)
ROMANIZE_CHUNK_S = 20.0  # window for re-priming the prompt (limits script drift)


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

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        if self.output_script == "roman":
            return self._transcribe_romanized(audio)
        return self._transcribe_native(audio)

    def _transcribe_native(self, audio: AudioData) -> TranscriptionResult:
        client = self._groq()
        metrics = STTMetrics.start(self.name, audio.duration_ms)
        resp = client.audio.transcriptions.create(
            file=("audio.wav", audio.to_wav_bytes()),
            model=self.model,
            response_format="json",
        )
        metrics.mark_final()
        text = (getattr(resp, "text", "") or "").strip()
        return TranscriptionResult(
            text=text, backend=self.name, metrics=metrics, raw=self._to_raw(resp)
        )

    def _transcribe_romanized(self, audio: AudioData) -> TranscriptionResult:
        client = self._groq()
        metrics = STTMetrics.start(self.name, audio.duration_ms)
        parts: list[str] = []
        last_resp: Any = None
        for i, chunk in enumerate(self._chunks(audio)):
            resp = client.audio.transcriptions.create(
                file=("audio.wav", chunk.to_wav_bytes()),
                model=self.model,
                response_format="json",
                language="en",
                prompt=self.romanize_prompt,
            )
            if i == 0:
                metrics.mark_first_token()
            last_resp = resp
            t = (getattr(resp, "text", "") or "").strip()
            if t:
                parts.append(t)
        metrics.mark_final()
        return TranscriptionResult(
            text=" ".join(parts).strip(),
            backend=self.name,
            metrics=metrics,
            raw=self._to_raw(last_resp) if last_resp is not None else {},
        )

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
