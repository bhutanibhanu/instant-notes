"""Run a set of STT backends over one audio clip and report the results."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from instant_notes.stt.base import AudioData, STTBackend
from instant_notes.bench.metrics import BenchResult, word_error_rate


def run_benchmark(
    audio: AudioData,
    backends: Sequence[STTBackend],
    reference: Optional[str] = None,
) -> list[BenchResult]:
    """Transcribe ``audio`` with each backend and collect a ``BenchResult``.

    A backend that raises does not abort the run — its failure is recorded as a
    result with ``text="<error: ...>"`` and infinite/None latencies so it sorts
    last and is obviously broken in the report.
    """
    results: list[BenchResult] = []
    for backend in backends:
        try:
            result = backend.transcribe(audio)
            metrics = result.metrics
            wer = (
                word_error_rate(reference, result.text)
                if reference is not None
                else None
            )
            results.append(
                BenchResult(
                    backend=backend.name,
                    text=result.text,
                    total_ms=metrics.total_ms,
                    first_token_ms=metrics.first_token_ms,
                    realtime_factor=metrics.realtime_factor,
                    wer=wer,
                )
            )
        except Exception as e:  # noqa: BLE001 - one bad backend must not kill the bench
            results.append(
                BenchResult(
                    backend=getattr(backend, "name", "unknown"),
                    text=f"<error: {e}>",
                    total_ms=math.inf,
                    first_token_ms=None,
                    realtime_factor=math.inf,
                    wer=None,
                )
            )
    return results


def format_report(results: Sequence[BenchResult]) -> str:
    """Render a markdown table sorted by total latency (ascending)."""
    rows = sorted(results, key=lambda r: r.total_ms)

    header = "| Backend | Total (ms) | First token (ms) | RTF | WER |"
    divider = "| --- | --- | --- | --- | --- |"
    lines = [header, divider]

    for r in rows:
        total = "inf" if math.isinf(r.total_ms) else f"{r.total_ms:.1f}"
        first = "-" if r.first_token_ms is None else f"{r.first_token_ms:.1f}"
        if math.isinf(r.realtime_factor):
            rtf = "inf"
        else:
            rtf = f"{r.realtime_factor:.2f}"
        wer = "-" if r.wer is None else f"{r.wer:.3f}"
        lines.append(f"| {r.backend} | {total} | {first} | {rtf} | {wer} |")

    return "\n".join(lines)
