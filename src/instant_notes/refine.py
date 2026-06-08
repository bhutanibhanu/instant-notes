"""Optional LLM cleanup pass for romanized-Hinglish transcripts.

Speech-to-text leaves residual spelling variants and occasional mishears
("text"→"take"). A cheap, fast Groq-Llama pass fixes these while keeping the
text in Roman script — measured WER ~0.45→0.30 (Groq STT) and ~0.29→0.26
(Sarvam STT) on the Hinglish sample. Degrades to the original text on any error.
"""

from __future__ import annotations

from instant_notes import llm

_CLEANUP_SYSTEM = (
    "You clean up Hinglish (romanized Hindi + English) speech-to-text "
    "transcripts. Fix obvious transcription errors and use natural, consistent "
    "romanized spelling. Keep the text in Roman script (NEVER Devanagari), keep "
    "English words in English, and preserve meaning, numbers, and named entities "
    "exactly. Return ONLY the corrected transcript with no preamble."
)


def clean_transcript(text: str, cfg) -> str:
    """Return an LLM-cleaned version of ``text``, or ``text`` unchanged if the
    cleanup is disabled, unavailable, or errors."""
    if not text or not getattr(cfg, "llm_cleanup", False):
        return text
    if not llm.available(cfg):
        return text
    cleaned = llm.chat(_CLEANUP_SYSTEM, text, cfg, temperature=0.0)
    return cleaned or text
