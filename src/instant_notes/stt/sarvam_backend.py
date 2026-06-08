"""Sarvam AI speech-to-text backend.

Uses the Sarvam REST API over ``httpx`` (a core dependency), so this module
imports cleanly without any optional SDK installed.
"""

from __future__ import annotations

from typing import Any

import httpx

from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
# Sarvam's sync endpoint rejects audio longer than 30s. We chunk anything longer
# into windows safely under that and concatenate. (Real voice notes are short, so
# this only kicks in for long dictation / the bench sample.)
SARVAM_MAX_CHUNK_S = 28.0


class SarvamBackend(STTBackend):
    """Sarvam AI cloud STT (``saarika`` family)."""

    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "saarika:v2",
        language_code: str = "unknown",
    ) -> None:
        self.api_key = api_key
        self.model = model
        # saarika:v2 requires language_code; "unknown" enables auto-detection.
        self.language_code = language_code
        # A single pooled client is reused across calls so the daemon doesn't
        # pay a fresh TLS handshake per note (latency).
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def is_available(self) -> bool:
        # Pure-httpx backend: always runnable as long as a key is configured.
        return bool(self.api_key)

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        metrics = STTMetrics.start(self.name, audio.duration_ms)
        chunks = self._split(audio)
        parts: list[str] = []
        last_payload: Any = {}
        for i, chunk in enumerate(chunks):
            payload = self._transcribe_chunk(chunk)
            if i == 0:
                metrics.mark_first_token()
            last_payload = payload
            if isinstance(payload, dict):
                t = (payload.get("transcript") or "").strip()
                if t:
                    parts.append(t)
        metrics.mark_final()

        text = " ".join(parts).strip()
        raw = last_payload if isinstance(last_payload, dict) else {"raw": last_payload}
        return TranscriptionResult(
            text=text,
            backend=self.name,
            metrics=metrics,
            raw=raw,
        )

    def _transcribe_chunk(self, audio: AudioData) -> Any:
        files = {"file": ("audio.wav", audio.to_wav_bytes(), "audio/wav")}
        data = {"model": self.model, "language_code": self.language_code}
        headers = {"api-subscription-key": self.api_key}
        resp = self._http().post(
            SARVAM_STT_URL, headers=headers, files=files, data=data
        )
        resp.raise_for_status()
        return self._parse_json(resp)

    @staticmethod
    def _split(audio: AudioData) -> list[AudioData]:
        """Split into <=SARVAM_MAX_CHUNK_S windows (sync API caps at 30s)."""
        win = int(SARVAM_MAX_CHUNK_S * audio.sample_rate)
        n = len(audio.samples)
        if n <= win:
            return [audio]
        return [
            AudioData(samples=audio.samples[i : i + win], sample_rate=audio.sample_rate)
            for i in range(0, n, win)
        ]

    @staticmethod
    def _parse_json(resp: Any) -> Any:
        try:
            return resp.json()
        except Exception:
            return {}
