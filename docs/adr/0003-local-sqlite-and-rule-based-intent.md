# ADR 0003 — Local SQLite storage + rule-based intent routing

- Status: accepted
- Date: 2026-06-08

## Context
Notes are personal and must be queryable offline and instantly ("show my notes",
"what did I note today", "summarize"). The query layer should not hard-depend on
a network LLM for basic retrieval, but should use Claude for summarization and
free-form questions.

## Decision
- **Storage:** SQLite with an FTS5 virtual table (`storage.py`). Local, zero-setup,
  fast full-text search, no server. Notes carry latency/backend metadata for
  later analysis.
- **Intent routing:** a rule-based classifier first (`intent.py`) — keywords/regex
  map to `show_recent | today | search | summarize | ask`. Retrieval intents are
  answered purely from SQLite (work with no API key). Only `summarize` and `ask`
  call Claude, and both degrade gracefully (local summary / "key not configured")
  when no `ANTHROPIC_API_KEY` is present.

## Deviation from the design doc
`design.md` §6 originally specified an Anthropic `router_model` for intent
routing. We intentionally route with rules instead (LLM reserved for summarize/
ask). This is the accepted decision; the `router_model` config key is retained
for a possible future LLM-router fallback but is not on the hot path today.

## Consequences
- The app is useful offline and with zero keys (capture + show/search/today).
- Claude is used where it adds real value (summarization, Q&A), not as a routing
  bottleneck — keeping the common path fast and free.
- A future enhancement could add an LLM-based router for fuzzier commands; the
  rule-based path stays as the fast/offline default.
