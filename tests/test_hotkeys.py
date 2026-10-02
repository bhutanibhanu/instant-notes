"""Hotkey validation — guards the 'space' vs '<space>' class of config bug."""

import pytest

# pynput needs a live input backend (an X display on Linux), so headless CI skips
# these instead of erroring at collection; they still run on the Mac.
HotKey = pytest.importorskip("pynput.keyboard", exc_type=ImportError).HotKey

from instant_notes.config import Config  # noqa: E402
from instant_notes.hotkeys import HotkeyListener  # noqa: E402


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
