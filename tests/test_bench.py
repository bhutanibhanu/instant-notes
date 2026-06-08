"""Tests for the benchmark harness and WER metric. No real models loaded."""

from __future__ import annotations

import numpy as np
import pytest

from instant_notes.bench.harness import format_report, run_benchmark
from instant_notes.bench.metrics import BenchResult, word_error_rate
from instant_notes.stt.base import (
    AudioData,
    STTBackend,
    STTMetrics,
    TranscriptionResult,
)


def _audio(seconds: float = 1.0, sample_rate: int = 16000) -> AudioData:
    n = int(seconds * sample_rate)
    return AudioData(samples=np.zeros(n, dtype=np.float32), sample_rate=sample_rate)


class FakeBackend(STTBackend):
    """Returns a fixed transcript with real, honestly-populated metrics."""

    def __init__(self, name: str = "fake", text: str = "the cat sat") -> None:
        self.name = name
        self._text = text

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        metrics = STTMetrics.start(self.name, audio.duration_ms)
        metrics.mark_first_token()
        metrics.mark_final()
        return TranscriptionResult(
            text=self._text, backend=self.name, metrics=metrics
        )


class ExplodingBackend(STTBackend):
    name = "boom"

    def transcribe(self, audio: AudioData) -> TranscriptionResult:
        raise RuntimeError("kaboom")


def test_word_error_rate():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    # one word wrong out of four -> 0.25
    assert word_error_rate("the cat sat down", "the cat sat up") == pytest.approx(0.25)
    # empty reference edge cases
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "anything") == 1.0


def test_run_benchmark_with_fake_backend():
    audio = _audio()
    backend = FakeBackend(name="fake", text="the cat sat")
    results = run_benchmark(audio, [backend], reference="the cat sat")

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, BenchResult)
    assert r.backend == "fake"
    assert r.text == "the cat sat"
    assert r.wer == 0.0
    assert r.total_ms >= 0.0
    assert r.first_token_ms is not None


def test_run_benchmark_no_reference_leaves_wer_none():
    results = run_benchmark(_audio(), [FakeBackend()], reference=None)
    assert results[0].wer is None


def test_format_report():
    results = run_benchmark(
        _audio(),
        [FakeBackend(name="alpha"), FakeBackend(name="beta")],
        reference="the cat sat",
    )
    report = format_report(results)
    assert isinstance(report, str)
    assert "alpha" in report
    assert "beta" in report
    assert "WER" in report


def test_run_benchmark_handles_backend_error():
    results = run_benchmark(_audio(), [ExplodingBackend()], reference="x")
    assert len(results) == 1
    r = results[0]
    assert r.backend == "boom"
    assert r.text.startswith("<error:")
    assert r.wer is None


def test_format_report_sorts_by_total_ms_with_error_last():
    good = FakeBackend(name="good")
    results = run_benchmark(_audio(), [ExplodingBackend(), good])
    report = format_report(results)
    # the working backend should appear before the errored one in the table
    assert report.index("good") < report.index("boom")


@pytest.mark.local_model
def test_faster_whisper_smoke():
    pytest.importorskip("faster_whisper")
    from instant_notes.stt.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(model="tiny")
    assert backend.is_available()
    result = backend.transcribe(_audio(seconds=1.0))
    assert isinstance(result.text, str)
