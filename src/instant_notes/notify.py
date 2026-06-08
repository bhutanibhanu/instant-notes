"""Notification / output layer for Instant Notes.

macOS notifications via ``osascript`` and clipboard-paste at the cursor via
``pbcopy`` + a simulated Cmd+V. Both are best-effort: they always print to
stdout (so the app works headless / in tests) and never raise.
"""

from __future__ import annotations

import subprocess


def _escape(s: str) -> str:
    """Escape characters that would break an AppleScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str, *, enabled: bool = True) -> None:
    """Display a macOS notification and always print the message to stdout.

    When ``enabled`` is False, only the stdout print happens. ``osascript``
    being missing (non-macOS / sandboxed) never raises.
    """
    # Always surface to stdout so this works headless and in tests.
    print(f"{title}: {message}")

    if not enabled:
        return

    script = (
        f'display notification "{_escape(message)}" '
        f'with title "{_escape(title)}"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
        )
    except Exception:
        # osascript missing or any other failure — degrade silently.
        pass


def paste_at_cursor(text: str) -> None:
    """Copy ``text`` to the clipboard and simulate Cmd+V at the cursor.

    Best-effort: never raises if ``pbcopy`` / ``osascript`` are unavailable.
    """
    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=False,
        )
    except Exception:
        return

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" '
                "using command down",
            ],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
