"""Backend factory. Concrete backends are imported lazily so that a missing
optional dependency (e.g. faster-whisper not installed) only errors if you
actually ask for that backend.

Backend constructor contracts (relied on by ``build_backend``):
  FasterWhisperBackend(model: str, compute_type: str)
  SarvamBackend(api_key: str, model: str)
  GroqBackend(api_key: str, model: str)
  DeepgramBackend(api_key: str, model: str)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instant_notes.config import Config
    from instant_notes.stt.base import STTBackend

BACKEND_NAMES = ["faster-whisper", "sarvam", "groq", "deepgram"]


def build_backend(name: str, cfg: Config) -> STTBackend:
    """Instantiate a backend by name using values from ``cfg``."""
    if name == "faster-whisper":
        from instant_notes.stt.faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend(
            model=cfg.whisper_model, compute_type=cfg.whisper_compute_type
        )
    if name == "sarvam":
        if not cfg.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY not set")
        from instant_notes.stt.sarvam_backend import SarvamBackend
        return SarvamBackend(api_key=cfg.sarvam_api_key, model=cfg.sarvam_model)
    if name == "groq":
        if not cfg.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")
        from instant_notes.stt.groq_backend import GroqBackend
        return GroqBackend(api_key=cfg.groq_api_key, model=cfg.groq_model)
    if name == "deepgram":
        if not cfg.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        from instant_notes.stt.deepgram_backend import DeepgramBackend
        return DeepgramBackend(api_key=cfg.deepgram_api_key, model=cfg.deepgram_model)
    raise ValueError(f"Unknown STT backend: {name!r}. Known: {BACKEND_NAMES}")


def available_backends(cfg: Config) -> list[str]:
    """Names of backends that can run right now (deps/keys present)."""
    return [name for name in BACKEND_NAMES if cfg.backend_available(name)]
