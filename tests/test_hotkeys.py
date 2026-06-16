"""Hotkey validation — guards the 'space' vs '<space>' class of config bug."""

import pytest
from pynput.keyboard import HotKey

from instant_notes.config import Config
from instant_notes.hotkeys import HotkeyListener


def test_default_hotkeys_parse():
    """Default config hotkeys must be valid pynput combos (regression)."""
    cfg = Config()
    HotKey.parse(cfg.capture_hotkey)  # raises if invalid
    HotKey.parse(cfg.command_hotkey)


def test_listener_rejects_bad_combo_with_clear_error():
    # bad: bare 'space' instead of '<space>'
    listener = HotkeyListener({"<cmd>+<shift>+space": lambda: None})
    with pytest.raises(ValueError, match="Invalid hotkey"):
        listener.start()
