"""Global hotkey listening via ``pynput``.

``pynput`` is imported lazily inside :meth:`HotkeyListener.start` so that this
module (and the rest of the app, including the test suite) imports cleanly on
machines without pynput / an accessible input backend installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class HotkeyListener:
    """Listen for a set of global hotkeys and fire callbacks.

    ``bindings`` maps a pynput hotkey string to a zero-arg callback, e.g.::

        HotkeyListener({"<cmd>+<shift>+space": on_capture})

    The underlying ``pynput.keyboard.GlobalHotKeys`` runs on its own thread.
    """

    def __init__(self, bindings: dict[str, Callable[[], None]]):
        self.bindings = bindings
        # pynput.keyboard.GlobalHotKeys (typed Any — pynput is a lazy/optional import)
        self._listener: Any = None

    def start(self) -> None:
        """Start listening on a background thread."""
        from pynput import keyboard  # lazy import

        self._listener = keyboard.GlobalHotKeys(self.bindings)
        self._listener.start()

    def stop(self) -> None:
        """Stop the listener thread if running."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def join(self) -> None:
        """Block on the listener thread until it stops."""
        if self._listener is not None:
            self._listener.join()
