"""Voice-command / LLM layer for Instant Notes.

``NoteAssistant`` classifies a spoken command into an intent and fulfills it
against the :class:`~instant_notes.storage.NoteStore`. Classification is
primarily rule-based (regex/keywords) so the assistant works with no API key.
Free-form questions and summaries are answered with Claude when a client is
available, and degrade gracefully when it is not.

The ``anthropic`` package is imported lazily — importing this module never
requires it. The client may be injected (for tests) or constructed on demand
from ``cfg.anthropic_api_key``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .storage import Note, NoteStore

# Intent constants -----------------------------------------------------------
SHOW_RECENT = "show_recent"
TODAY = "today"
SEARCH = "search"
SUMMARIZE = "summarize"
ASK = "ask"
UNKNOWN = "unknown"


@dataclass
class CommandResult:
    intent: str
    response_text: str
    notes: list[Note] = field(default_factory=list)


# Regexes for extracting a search query from natural phrasing.
_SEARCH_PATTERNS = [
    re.compile(r"\bsearch(?:\s+for)?\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bfind\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bnotes?\s+about\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bsearch\s+(.+)$", re.IGNORECASE),
]

_NO_CLIENT_MSG = (
    "Claude API key not configured — can't answer free-form questions yet."
)


def _format_notes(notes: list[Note]) -> str:
    """Render notes as a numbered list of their text."""
    return "\n".join(f"{i}. {n.text}" for i, n in enumerate(notes, start=1))


def _strip_query(text: str) -> str:
    return text.strip().strip("?.!,").strip()


class NoteAssistant:
    """Classify and fulfill spoken note commands."""

    def __init__(
        self,
        store: NoteStore,
        cfg: Config,
        client=None,
    ):
        self.store = store
        self.cfg = cfg
        self._client = client
        # Track whether we've already attempted lazy construction so we don't
        # retry an import that failed.
        self._client_attempted = client is not None

    # -- client ------------------------------------------------------------
    @property
    def client(self):
        """Return the Claude client, lazily constructing it if possible.

        ``anthropic`` is imported here so module import never requires the
        package or a key. Returns ``None`` when no key/package is available.
        """
        if self._client is None and not self._client_attempted:
            self._client_attempted = True
            api_key = getattr(self.cfg, "anthropic_api_key", None)
            if api_key:
                try:
                    import anthropic  # lazy import

                    self._client = anthropic.Anthropic(api_key=api_key)
                except Exception:
                    self._client = None
        return self._client

    # -- routing -----------------------------------------------------------
    def handle_command(self, text: str) -> CommandResult:
        """Classify ``text`` into an intent and produce a result.

        Never raises for normal flow; Claude/network errors degrade to a
        helpful message.
        """
        raw = text or ""
        lowered = raw.lower()

        # Rule-based classification. Order matters: "today" and "summarize"
        # are checked before the generic show/search fallbacks.
        if re.search(r"\btoday'?s?\b", lowered) and "summar" not in lowered:
            return self._handle_today()

        if "summar" in lowered:
            return self._handle_summarize(raw, lowered)

        query = self._extract_search_query(raw)
        if query is not None:
            return self._handle_search(query)

        if re.search(r"\b(show|list|recent|my notes|what notes)\b", lowered):
            return self._handle_show_recent()

        # Anything else is a free-form question about the notes.
        return self._handle_ask(raw)

    # -- query extraction --------------------------------------------------
    def _extract_search_query(self, text: str) -> str | None:
        for pat in _SEARCH_PATTERNS:
            m = pat.search(text)
            if m:
                q = _strip_query(m.group(1))
                if q:
                    return q
        return None

    # -- intent handlers ---------------------------------------------------
    def _handle_show_recent(self) -> CommandResult:
        notes = self.store.recent(10) if self.store else []
        if not notes:
            return CommandResult(SHOW_RECENT, "No notes yet.", [])
        return CommandResult(SHOW_RECENT, _format_notes(notes), notes)

    def _handle_today(self) -> CommandResult:
        notes = self.store.today() if self.store else []
        if not notes:
            return CommandResult(TODAY, "No notes today.", [])
        return CommandResult(TODAY, _format_notes(notes), notes)

    def _handle_search(self, query: str) -> CommandResult:
        notes = self.store.search(query) if self.store else []
        if not notes:
            return CommandResult(SEARCH, f"No matches for '{query}'.", [])
        return CommandResult(SEARCH, _format_notes(notes), notes)

    def _handle_summarize(self, raw: str, lowered: str) -> CommandResult:
        if self.store is None:
            notes: list[Note] = []
        elif re.search(r"\btoday'?s?\b", lowered):
            notes = self.store.today()
        else:
            notes = self.store.recent(50)

        if not notes:
            return CommandResult(SUMMARIZE, "No notes to summarize.", [])

        client = self.client
        if client is not None:
            summary = self._call_claude(
                "Summarize the following voice notes concisely. Group related "
                "items and surface anything that looks like a task or "
                "reminder. Be brief.\n\n" + _format_notes(notes)
            )
            if summary is not None:
                return CommandResult(SUMMARIZE, summary, notes)
            # fall through to local summary on error

        return CommandResult(SUMMARIZE, self._local_summary(notes), notes)

    def _handle_ask(self, question: str) -> CommandResult:
        notes = self.store.recent(50) if self.store else []
        client = self.client
        if client is None:
            return CommandResult(ASK, _NO_CLIENT_MSG, notes)

        prompt = (
            "You are answering a question about the user's personal voice "
            "notes. Use ONLY the notes below; if the answer isn't there, say "
            "so. Be concise.\n\n"
            f"Question: {question}\n\nNotes:\n{_format_notes(notes)}"
        )
        answer = self._call_claude(prompt)
        if answer is None:
            return CommandResult(
                ASK,
                "Sorry — I couldn't reach Claude to answer that right now.",
                notes,
            )
        return CommandResult(ASK, answer, notes)

    # -- helpers -----------------------------------------------------------
    def _local_summary(self, notes: list[Note]) -> str:
        """Fallback summary used when no Claude client is available."""
        n = len(notes)
        joined = " ".join(n_.text for n_ in notes)
        snippet = joined[:280] + ("…" if len(joined) > 280 else "")
        return f"You have {n} note{'s' if n != 1 else ''}. {snippet}".strip()

    def _call_claude(self, prompt: str) -> str | None:
        """Call the Messages API; return text or ``None`` on any error."""
        client = self.client
        if client is None:
            return None
        try:
            resp = client.messages.create(
                model=self.cfg.answer_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._extract_text(resp)
        except Exception:
            return None

    @staticmethod
    def _extract_text(resp) -> str | None:
        content = getattr(resp, "content", None)
        if not content:
            return None
        # content is a list of blocks; concatenate text blocks.
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts).strip()
        return None
