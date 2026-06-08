"""Benchmark harness: compare STT backends on latency and accuracy."""

from instant_notes.bench.harness import format_report, run_benchmark
from instant_notes.bench.metrics import BenchResult, word_error_rate

__all__ = [
    "BenchResult",
    "word_error_rate",
    "run_benchmark",
    "format_report",
]
