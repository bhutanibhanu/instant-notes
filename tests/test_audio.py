import numpy as np

from instant_notes.audio import trim_silence
from instant_notes.stt.base import AudioData


def _tone(seconds: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_silence_removes_leading_trailing():
    sr = 16000
    silence = np.zeros(sr, np.float32)  # 1s each side
    speech = _tone(1.0, sr)
    samples = np.concatenate([silence, speech, silence])
    audio = AudioData(samples=samples, sample_rate=sr)
    trimmed = trim_silence(audio, pad_ms=50)
    # ~1s of speech + small padding, well under the original 3s
    assert len(trimmed.samples) < len(audio.samples)
    assert 0.9 * sr <= len(trimmed.samples) <= 1.4 * sr


def test_trim_silence_all_quiet_returns_original():
    sr = 16000
    audio = AudioData(samples=np.zeros(sr, np.float32), sample_rate=sr)
    trimmed = trim_silence(audio)
    assert len(trimmed.samples) == len(audio.samples)  # don't return empty


def test_trim_silence_empty():
    audio = AudioData(samples=np.zeros(0, np.float32), sample_rate=16000)
    assert len(trim_silence(audio).samples) == 0
