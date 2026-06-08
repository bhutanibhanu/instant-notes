"""Microphone capture. Toggle-to-record: ``start()`` then ``stop()`` returns an
``AudioData`` (16 kHz mono float32) ready for any STT backend.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from instant_notes.stt.base import AudioData

TARGET_SAMPLE_RATE = 16_000  # what every STT model here expects


class AudioRecorder:
    """Non-blocking recorder backed by a sounddevice input stream. Frames are
    appended on the audio thread; ``stop()`` concatenates them.

    ``sounddevice`` is imported lazily so the module (and the test suite) load
    without PortAudio present.
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._stream: Any = None  # sounddevice.InputStream (lazy/optional import)
        self._lock = threading.Lock()
        self._recording = False
        self._on_chunk: Callable[[np.ndarray], None] | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _callback(self, indata, frames, time_info, status):
        # Runs on the PortAudio thread; keep it cheap.
        with self._lock:
            self._frames.append(indata.copy())
        # Live streaming: hand a mono float32 copy to the consumer. The
        # callback must be non-blocking (it enqueues), so we never stall the
        # audio thread.
        if self._on_chunk is not None:
            self._on_chunk(indata.copy().reshape(-1))

    def start(self, on_chunk: Callable[[np.ndarray], None] | None = None) -> None:
        """Begin capturing. If ``on_chunk`` is given, each captured block is
        also delivered to it live (mono float32) for streaming transcription;
        ``stop()`` still returns the full :class:`AudioData` either way."""
        if self._recording:
            return
        import sounddevice as sd  # lazy

        self._on_chunk = on_chunk
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._recording = True

    def stop(self) -> AudioData:
        if not self._recording:
            return AudioData(samples=np.zeros(0, dtype=np.float32),
                             sample_rate=self.sample_rate)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False
        self._on_chunk = None
        with self._lock:
            frames = self._frames
            self._frames = []
        if frames:
            data = np.concatenate(frames, axis=0)
            if data.ndim > 1:  # downmix to mono
                data = data.mean(axis=1)
        else:
            data = np.zeros(0, dtype=np.float32)
        return AudioData(samples=data.astype(np.float32),
                         sample_rate=self.sample_rate)


def load_wav(path: str) -> AudioData:
    """Read a WAV file into ``AudioData`` (mono, resampled to 16 kHz). Used by
    the benchmark harness to replay a sample through every backend."""
    import wave

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        samples = _resample(samples, sample_rate, TARGET_SAMPLE_RATE)
    return AudioData(samples=samples.astype(np.float32),
                     sample_rate=TARGET_SAMPLE_RATE)


def trim_silence(
    audio: AudioData,
    threshold: float = 0.01,
    frame_ms: float = 30.0,
    pad_ms: float = 120.0,
) -> AudioData:
    """Trim leading/trailing near-silence using per-frame RMS energy.

    Notes captured by hotkey toggle have dead air at both ends (press → speak →
    press); trimming it sends fewer bytes (lower latency) and avoids Whisper
    hallucinating words over silence (better accuracy). A small ``pad_ms`` is
    kept around the speech so we don't clip onsets. Returns the original audio
    unchanged if it's all below threshold (don't return an empty clip).
    """
    n = len(audio.samples)
    if n == 0:
        return audio
    frame = max(1, int(frame_ms / 1000.0 * audio.sample_rate))
    n_frames = n // frame
    if n_frames < 2:
        return audio
    trimmed = audio.samples[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    voiced = np.where(rms >= threshold)[0]
    if len(voiced) == 0:
        return audio  # all quiet — keep as-is rather than returning nothing
    pad = int(pad_ms / 1000.0 * audio.sample_rate)
    start = max(0, voiced[0] * frame - pad)
    end = min(n, (voiced[-1] + 1) * frame + pad)
    return AudioData(samples=audio.samples[start:end], sample_rate=audio.sample_rate)


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    duration = len(samples) / src_rate
    n_dst = int(round(duration * dst_rate))
    src_t = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    return np.interp(dst_t, src_t, samples)
