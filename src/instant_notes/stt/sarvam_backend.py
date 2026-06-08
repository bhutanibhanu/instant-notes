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
        wav_bytes = audio.to_wav_bytes()
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "language_code": self.language_code}
        headers = {"api-subscription-key": self.api_key}

        metrics = STTMetrics.start(self.name, audio.duration_ms)
        resp = self._http().post(
            SARVAM_STT_URL, headers=headers, files=files, data=data
        )
        resp.raise_for_status()
        payload = self._parse_json(resp)
        metrics.mark_final()

        text = ""
        if isinstance(payload, dict):
            text = payload.get("transcript") or ""
        text = (text or "").strip()

        return TranscriptionResult(
            text=text,
            backend=self.name,
            metrics=metrics,
            raw=payload if isinstance(payload, dict) else {"raw": payload},
        )

    @staticmethod
    def _parse_json(resp: Any) -> Any:
        try:
            return resp.json()
        except Exception:
            return {}
