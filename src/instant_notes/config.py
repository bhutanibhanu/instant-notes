"""Configuration: layered ``config.toml`` (behavior) + ``.env`` (secrets).

Lookup order for files: explicit path → ``./config.toml`` → ``~/.config/instant-notes/config.toml``.
Missing files are fine; defaults below apply. Secrets are read from the
environment (``.env`` is loaded if present).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

DEFAULT_CONFIG_PATHS = [
    Path("config.toml"),
    Path.home() / ".config" / "instant-notes" / "config.toml",
]
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "instant-notes" / "notes.db"


@dataclass
class Config:
    # hotkeys (pynput format, e.g. "<cmd>+<shift>+space")
    capture_hotkey: str = "<cmd>+<shift>+space"
    command_hotkey: str = "<cmd>+<shift>+a"

    # STT
    default_backend: str = "faster-whisper"
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"  # cpu-friendly default
    sarvam_model: str = "saarika:v2"
    groq_model: str = "whisper-large-v3-turbo"
    deepgram_model: str = "nova-3"

    # LLM (intent routing + answering)
    router_model: str = "claude-haiku-4-5"
    answer_model: str = "claude-opus-4-8"

    # behavior
    paste_at_cursor: bool = False
    notify: bool = True
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    # secrets (from env)
    sarvam_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    def backend_available(self, name: str) -> bool:
        return {
            "faster-whisper": True,
            "sarvam": bool(self.sarvam_api_key),
            "groq": bool(self.groq_api_key),
            "deepgram": bool(self.deepgram_api_key),
        }.get(name, False)


def _first_existing(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for p in DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    return None


def load_config(path: Optional[str] = None) -> Config:
    load_dotenv()  # populate os.environ from a local .env if present
    cfg = Config()

    toml_path = _first_existing(path)
    if toml_path:
        with open(toml_path, "rb") as fh:
            data = tomllib.load(fh)
        for section in ("hotkeys", "stt", "llm", "behavior"):
            for key, value in (data.get(section) or {}).items():
                if hasattr(cfg, key) and key != "db_path":
                    setattr(cfg, key, value)
        behavior = data.get("behavior") or {}
        if "db_path" in behavior:
            cfg.db_path = Path(behavior["db_path"]).expanduser()

    cfg.sarvam_api_key = os.getenv("SARVAM_API_KEY") or cfg.sarvam_api_key
    cfg.groq_api_key = os.getenv("GROQ_API_KEY") or cfg.groq_api_key
    cfg.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY") or cfg.deepgram_api_key
    cfg.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or cfg.anthropic_api_key
    return cfg
