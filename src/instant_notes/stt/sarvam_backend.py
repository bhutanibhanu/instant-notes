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
SARVAM_TRANSLITERATE_URL = "https://api.sarvam.ai/transliterate"
# Sarvam's sync endpoint rejects audio longer than 30s. We chunk anything longer
# into windows safely under that and concatenate. (Real voice notes are short, so
# this only kicks in for long dictation / the bench sample.)
SARVAM_MAX_CHUNK_S = 28.0
# Transliterate endpoint input cap (chars); we split longer text on spaces.
SARVAM_TRANSLIT_MAX_CHARS = 900


class SarvamBackend(STTBackend):
    """Sarvam AI cloud STT (``saarika`` family).

    ``output_script="roman"`` pipes the (Devanagari) transcript through Sarvam's
    transliterate API with ``spoken_form=True`` to produce natural romanized
    Hinglish — proper schwa deletion ("naam", "aur") and English words recovered.
    """

    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "saarika:v2.5",
        language_code: str = "unknown",
        output_script: str = "native",
    ) -> None:
        self.api_key = api_key
        self.model = model
        # saarika requires language_code; "unknown" enables auto-detection.
        self.language_code = language_code
        self.output_script = output_script
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
        # STT chunks are independent → run concurrently (latency = slowest, not sum).
        payloads = self._parallel_map(self._transcribe_chunk, chunks)
        metrics.mark_first_token()
        last_payload: Any = payloads[-1] if payloads else {}
        parts = [
            (p.get("transcript") or "").strip()
            for p in payloads
            if isinstance(p, dict) and (p.get("transcript") or "").strip()
        ]

        text = " ".join(parts).strip()
        if self.output_script == "roman" and text:
            text = self._romanize(text)
        metrics.mark_final()

        raw = last_payload if isinstance(last_payload, dict) else {"raw": last_payload}
        return TranscriptionResult(
            text=text,
            backend=self.name,
            metrics=metrics,
            raw=raw,
        )

    def _romanize(self, text: str) -> str:
        """Transliterate Devanagari Hinglish -> natural romanized Hinglish via
        Sarvam's transliterate API (chunked on spaces, run concurrently)."""
        chunks = self._split_text(text, SARVAM_TRANSLIT_MAX_CHARS)
        out = self._parallel_map(self._transliterate_chunk, chunks)
        return " ".join(p for p in out if p).strip()

    def _transliterate_chunk(self, chunk: str) -> str:
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        resp = self._http().post(
            SARVAM_TRANSLITERATE_URL,
            headers=headers,
            json={
                "input": chunk,
                "source_language_code": "hi-IN",
                "target_language_code": "en-IN",
                "spoken_form": True,
            },
        )
        resp.raise_for_status()
        payload = self._parse_json(resp)
        if not isinstance(payload, dict):
            return ""
        return payload.get("transliterated_text") or ""

    @staticmethod
    def _parallel_map(fn, items: list):
        """Map fn over items concurrently, preserving order. httpx.Client is
        thread-safe, so the pooled client is shared across worker threads."""
        if len(items) <= 1:
            return [fn(x) for x in items]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
            return list(pool.map(fn, items))

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        buf = ""
        for w in words:
            if len(buf) + len(w) + 1 > max_chars:
                if buf:
                    chunks.append(buf)
                buf = w
            else:
                buf = (buf + " " + w).strip()
        if buf:
            chunks.append(buf)
        return chunks or [text]

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
