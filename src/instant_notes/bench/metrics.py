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


# Approximate published list prices (USD per minute of audio) for a rough
# cost-per-note estimate in the bench report. Local models are free. These are
# ballpark figures for comparison only — verify against current provider pricing.
COST_PER_MINUTE_USD: dict[str, float] = {
    "faster-whisper": 0.0,    # local, no API cost
    "groq": 0.0067,           # whisper-large-v3-turbo (~$0.04/hr)
    "deepgram": 0.0043,       # nova-3 streaming (~$0.0258/min tier varies)
    "sarvam": 0.005,          # approximate
}


def estimate_cost_usd(backend: str, audio_duration_ms: float) -> float | None:
    """Rough cost for transcribing this clip with ``backend`` (None if unknown)."""
    rate = COST_PER_MINUTE_USD.get(backend)
    if rate is None:
        return None
    return rate * (audio_duration_ms / 1000.0 / 60.0)


@dataclass
class BenchResult:
    backend: str
    text: str
    total_ms: float
    first_token_ms: float | None
    realtime_factor: float
    wer: float | None = None
    stop_to_final_ms: float | None = None
    cost_usd: float | None = None
