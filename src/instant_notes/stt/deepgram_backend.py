"""Deepgram speech-to-text backend (nova-3).

Uses the ``deepgram-sdk`` v3 client, imported lazily so this module imports
cleanly even when the package is not installed. Supports both batch
(pre-recorded) transcription and true live websocket streaming.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

from instant_notes.stt.base import (
    AudioData,
    StreamingSession,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)


class DeepgramBackend(STTBackend):
    """Deepgram cloud STT (pre-recorded + live streaming)."""

    name = "deepgram"

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None  # cached DeepgramClient, reused across calls

    def _ensure_client(self):
        """Construct and cache the DeepgramClient on first use."""
        if self._client is None:
            from deepgram import DeepgramClient

            self._client = DeepgramClient(self.api_key)
        return self._client

    def is_available(self) -> bool:
        try:
            import deepgram  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def supports_streaming(self) -> bool:
        try:
            import deepgram  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        from deepgram import PrerecordedOptions

        wav_bytes = audio.to_wav_bytes()
        dg = self._ensure_client()
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

    def open_stream(self, sample_rate: int) -> StreamingSession:
        from deepgram import LiveOptions

        dg = self._ensure_client()
        options = LiveOptions(
            model=self.model,
            language="en",
            encoding="linear16",
            sample_rate=sample_rate,
            channels=1,
            smart_format=True,
            interim_results=True,
        )
        return _DeepgramStreamingSession(dg, options, self.name)

    @staticmethod
    def _extract_transcript(resp: Any) -> str:
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


def _float32_to_int16_bytes(chunk: np.ndarray) -> bytes:
    """Convert mono float32 [-1, 1] to 16-bit little-endian PCM bytes."""
    clipped = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class _DeepgramStreamingSession(StreamingSession):
    """Live transcription over Deepgram's websocket.

    Audio fed by the recorder is enqueued and shipped to the socket by a
    dedicated worker thread, so :meth:`feed` never blocks the audio callback.
    Transcript events accumulate FINAL segments and track the latest interim.
    """

    def __init__(self, client, options, backend_name: str) -> None:
        self._backend_name = backend_name
        self._options = options
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._finals: list[str] = []
        self._partial: str = ""
        self._lock = threading.Lock()
        self._fed_samples = 0
        self._sample_rate = getattr(options, "sample_rate", 16_000) or 16_000
        self._closed = False
        self._final_event = threading.Event()

        # Metrics start the moment the stream opens (audio_duration filled in
        # at finish() from how much we actually fed).
        self._metrics = STTMetrics.start(backend_name, 0.0)

        self._connection = client.listen.websocket.v("1")
        self._wire_handlers()

        # Start the connection, then the sender worker.
        started = self._connection.start(options)
        if started is False:  # SDK returns False on failure
            raise RuntimeError("Deepgram websocket failed to start")
        self._worker = threading.Thread(target=self._sender, daemon=True)
        self._worker.start()

    def _wire_handlers(self) -> None:
        from deepgram import LiveTranscriptionEvents

        def on_transcript(_client, result, *args, **kwargs):  # noqa: ANN001
            self._metrics.mark_first_token()
            try:
                alt = result.channel.alternatives[0]
                text = (alt.transcript or "").strip()
                is_final = bool(getattr(result, "is_final", False))
            except (AttributeError, IndexError, TypeError):
                return
            if not text:
                return
            if is_final:
                with self._lock:
                    self._finals.append(text)
                    self._partial = ""
            else:
                with self._lock:
                    self._partial = text

        def on_utterance_end(_client, *args, **kwargs):  # noqa: ANN001
            self._final_event.set()

        self._connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        # Best-effort: not all SDK versions emit these, but wiring is harmless.
        try:
            self._connection.on(
                LiveTranscriptionEvents.UtteranceEnd, on_utterance_end
            )
        except Exception:
            pass

    def _sender(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._connection.send(item)
            except Exception:
                # Drop the chunk on transient send errors; finish() still
                # returns whatever finals accumulated.
                pass

    def feed(self, chunk: np.ndarray) -> None:
        if self._closed:
            return
        arr = np.asarray(chunk, dtype=np.float32)
        self._fed_samples += arr.size
        self._queue.put(_float32_to_int16_bytes(arr))

    def latest_partial(self) -> str:
        with self._lock:
            return self._partial

    def finish(self) -> TranscriptionResult:
        self._closed = True
        self._metrics.audio_duration_ms = (
            1000.0 * self._fed_samples / self._sample_rate
            if self._sample_rate
            else 0.0
        )
        try:
            # Stop the sender, then ask Deepgram to flush + close.
            self._queue.put(None)
            self._worker.join(timeout=3.0)
            try:
                self._connection.finalize()
            except Exception:
                pass
            # Wait briefly for trailing finals to arrive.
            self._final_event.wait(timeout=3.0)
            try:
                self._connection.finish()
            except Exception:
                pass
        except Exception:
            pass

        self._metrics.mark_final()
        with self._lock:
            text = " ".join(p for p in self._finals if p).strip()
            partial = self._partial
        # If no finals arrived but we have a partial, fall back to it.
        if not text and partial:
            text = partial.strip()
        return TranscriptionResult(
            text=text,
            backend=self._backend_name,
            metrics=self._metrics,
            raw={"finals": list(self._finals)},
        )
