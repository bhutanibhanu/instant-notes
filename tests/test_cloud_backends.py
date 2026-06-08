"""Unit tests for the 3 cloud STT backends. No real network calls — the HTTP
client / SDKs are faked via monkeypatch."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from instant_notes.stt.base import AudioData
from instant_notes.stt.deepgram_backend import DeepgramBackend
from instant_notes.stt.groq_backend import GroqBackend
from instant_notes.stt.sarvam_backend import SarvamBackend


@pytest.fixture
def audio() -> AudioData:
    return AudioData(samples=np.zeros(16000, np.float32), sample_rate=16000)


def test_groq_romanize_chunks_long_audio():
    """Roman mode chunks long audio (~20s windows) to limit Whisper script drift."""
    from instant_notes.stt.groq_backend import ROMANIZE_CHUNK_S, GroqBackend

    sr = 16000
    long_audio = AudioData(samples=np.zeros(int(132 * sr), np.float32), sample_rate=sr)
    chunks = GroqBackend._chunks(long_audio)
    assert len(chunks) == 7  # ceil(132 / 20)
    assert all(len(c.samples) <= ROMANIZE_CHUNK_S * sr for c in chunks)
    short = AudioData(samples=np.zeros(int(10 * sr), np.float32), sample_rate=sr)
    assert len(GroqBackend._chunks(short)) == 1


def test_groq_default_native_mode():
    b = GroqBackend(api_key="k")
    assert b.output_script == "native"
    b2 = GroqBackend(api_key="k", output_script="roman")
    assert b2.romanize_prompt  # has a default romanize prompt


def test_sarvam_split_chunks_long_audio():
    """Audio over Sarvam's 30s sync limit is split into <=28s windows."""
    from instant_notes.stt.sarvam_backend import SARVAM_MAX_CHUNK_S, SarvamBackend

    sr = 16000
    long_audio = AudioData(samples=np.zeros(int(132 * sr), np.float32), sample_rate=sr)
    chunks = SarvamBackend._split(long_audio)
    assert len(chunks) == 5  # ceil(132 / 28)
    assert all(len(c.samples) <= SARVAM_MAX_CHUNK_S * sr for c in chunks)
    # short audio is a single chunk (no splitting)
    short = AudioData(samples=np.zeros(int(10 * sr), np.float32), sample_rate=sr)
    assert len(SarvamBackend._split(short)) == 1


# --------------------------------------------------------------------------- #
# Sarvam — fake httpx.Client.post
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_sarvam_transcribe(audio, monkeypatch):
    captured = {}

    def fake_post(self, url, **kwargs):  # noqa: ANN001
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["files"] = kwargs.get("files")
        captured["data"] = kwargs.get("data")
        return _FakeResponse({"transcript": "hello world"})

    monkeypatch.setattr("httpx.Client.post", fake_post)

    backend = SarvamBackend(api_key="sk-test", model="saarika:v2")
    result = backend.transcribe(audio)

    assert result.text == "hello world"
    assert result.backend == "sarvam"
    assert result.metrics.total_ms >= 0
    assert result.metrics.backend == "sarvam"
    # contract: key header + model field sent
    assert captured["headers"]["api-subscription-key"] == "sk-test"
    assert captured["data"]["model"] == "saarika:v2"
    assert captured["files"]["file"][0] == "audio.wav"


def test_sarvam_defensive_bad_json(audio, monkeypatch):
    class _BadResp:
        def json(self):
            raise ValueError("not json")

        def raise_for_status(self):
            return None

    monkeypatch.setattr("httpx.Client.post", lambda self, url, **kw: _BadResp())
    backend = SarvamBackend(api_key="sk-test")
    result = backend.transcribe(audio)
    assert result.text == ""
    assert result.backend == "sarvam"


def test_sarvam_is_available():
    assert SarvamBackend(api_key="x").is_available() is True
    assert SarvamBackend(api_key="").is_available() is False


# --------------------------------------------------------------------------- #
# Groq — fake the lazily-imported ``groq`` module
# --------------------------------------------------------------------------- #
class _FakeGroqResult:
    text = "hi"


class _FakeGroqClient:
    def __init__(self, api_key=None):  # noqa: ANN001
        self.api_key = api_key
        self.audio = types.SimpleNamespace(
            transcriptions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):  # noqa: ANN003
        self.create_kwargs = kwargs
        return _FakeGroqResult()


def _install_fake_groq(monkeypatch):
    fake_mod = types.ModuleType("groq")
    fake_mod.Groq = _FakeGroqClient
    monkeypatch.setitem(sys.modules, "groq", fake_mod)


def test_groq_transcribe(audio, monkeypatch):
    _install_fake_groq(monkeypatch)

    backend = GroqBackend(api_key="gsk-test", model="whisper-large-v3-turbo")
    result = backend.transcribe(audio)

    assert result.text == "hi"
    assert result.backend == "groq"
    assert result.metrics.total_ms >= 0


def test_groq_is_available(monkeypatch):
    _install_fake_groq(monkeypatch)
    assert GroqBackend(api_key="x").is_available() is True
    # Missing SDK -> not available.
    monkeypatch.setitem(sys.modules, "groq", None)
    assert GroqBackend(api_key="x").is_available() is False


# --------------------------------------------------------------------------- #
# Deepgram — fake the lazily-imported ``deepgram`` module
# --------------------------------------------------------------------------- #
def _make_dg_response(transcript: str):
    alt = types.SimpleNamespace(transcript=transcript)
    channel = types.SimpleNamespace(alternatives=[alt])
    results = types.SimpleNamespace(channels=[channel])
    return types.SimpleNamespace(results=results)


class _FakeDeepgramClient:
    def __init__(self, api_key=None):  # noqa: ANN001
        self.api_key = api_key
        rest_v = types.SimpleNamespace(transcribe_file=self._transcribe)
        self.listen = types.SimpleNamespace(
            rest=types.SimpleNamespace(v=lambda _ver: rest_v)
        )

    def _transcribe(self, source, options):  # noqa: ANN001
        self.source = source
        self.options = options
        return _make_dg_response("yo")


class _FakePrerecordedOptions:
    def __init__(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs


def _install_fake_deepgram(monkeypatch):
    fake_mod = types.ModuleType("deepgram")
    fake_mod.DeepgramClient = _FakeDeepgramClient
    fake_mod.PrerecordedOptions = _FakePrerecordedOptions
    monkeypatch.setitem(sys.modules, "deepgram", fake_mod)


def test_deepgram_transcribe(audio, monkeypatch):
    _install_fake_deepgram(monkeypatch)

    backend = DeepgramBackend(api_key="dg-test", model="nova-3")
    result = backend.transcribe(audio)

    assert result.text == "yo"
    assert result.backend == "deepgram"
    assert result.metrics.total_ms >= 0


def test_deepgram_extract_defensive():
    # Missing/garbled shape -> empty string, no exception.
    assert DeepgramBackend._extract_transcript(object()) == ""
    assert DeepgramBackend._extract_transcript({"results": {}}) == ""


def test_deepgram_is_available(monkeypatch):
    _install_fake_deepgram(monkeypatch)
    assert DeepgramBackend(api_key="x").is_available() is True
    monkeypatch.setitem(sys.modules, "deepgram", None)
    assert DeepgramBackend(api_key="x").is_available() is False
