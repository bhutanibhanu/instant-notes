"""Accuracy metric (WER) and the per-backend result record."""

from __future__ import annotations

from dataclasses import dataclass


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate between ``reference`` and ``hypothesis``.

    Both are lowercased and stripped first. If the reference is empty we treat
    a matching empty hypothesis as perfect (0.0) and any non-empty hypothesis
    as fully wrong (1.0), since WER is undefined with no reference words.
    """
    ref = (reference or "").lower().strip()
    hyp = (hypothesis or "").lower().strip()

    if not ref:
        return 0.0 if not hyp else 1.0

    import jiwer

    try:
        return float(jiwer.wer(ref, hyp))
    except Exception:
        # jiwer can raise on pathological inputs; fall back to a sane value.
        return 0.0 if ref == hyp else 1.0


@dataclass
class BenchResult:
    backend: str
    text: str
    total_ms: float
    first_token_ms: float | None
    realtime_factor: float
    wer: float | None = None
