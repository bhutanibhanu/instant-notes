"""Run a set of STT backends over one audio clip and report the results."""

from __future__ import annotations

import math
from collections.abc import Sequence

from instant_notes.bench.metrics import (
    BenchResult,
    estimate_cost_usd,
    word_error_rate,
)
from instant_notes.stt.base import AudioData, STTBackend


def run_benchmark(
    audio: AudioData,
    backends: Sequence[STTBackend],
    reference: str | None = None,
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
                    stop_to_final_ms=metrics.stop_to_final_ms,
                    cost_usd=estimate_cost_usd(
                        backend.name, metrics.audio_duration_ms
                    ),
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

    header = (
        "| Backend | Total (ms) | First token (ms) | Stop→text (ms) "
        "| RTF | WER | Est. cost (USD) |"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, divider]

    for r in rows:
        total = "inf" if math.isinf(r.total_ms) else f"{r.total_ms:.1f}"
        first = "-" if r.first_token_ms is None else f"{r.first_token_ms:.1f}"
        stop = "-" if r.stop_to_final_ms is None else f"{r.stop_to_final_ms:.1f}"
        if math.isinf(r.realtime_factor):
            rtf = "inf"
        else:
            rtf = f"{r.realtime_factor:.2f}"
        wer = "-" if r.wer is None else f"{r.wer:.3f}"
        cost = "-" if r.cost_usd is None else f"${r.cost_usd:.5f}"
        lines.append(
            f"| {r.backend} | {total} | {first} | {stop} | {rtf} | {wer} | {cost} |"
        )

    # Transcripts — so you can eyeball *what* each backend actually heard,
    # not just the numbers.
    lines.append("")
    lines.append("### Transcripts")
    for r in rows:
        lines.append(f"- **{r.backend}**: {r.text}")

    return "\n".join(lines)
