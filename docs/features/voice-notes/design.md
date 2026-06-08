# Design: Instant Notes (voice-notes MVP)

> Status: scoped 2026-06-08. Synthesized from user brief + clarifying answers.
> Assumptions are marked **[ASSUMPTION]** — override any of these and I'll adjust.

## 1. Goal

A macOS background daemon that turns speech into notes instantly, with a
Wispr-Flow-class latency/accuracy bar. Two interactions, both hotkey-driven:

1. **Capture** — press hotkey to start recording, press again to stop. Audio is
   transcribed and saved as a timestamped note.
2. **Command/Query** — press a second hotkey, speak a request ("show my notes",
   "summarize my notes", "what did I note today"). The spoken command is
   transcribed, its intent is classified, and the app answers using the stored
   notes + the Claude API.

## 2. Non-negotiables

- **Low latency.** Stop-talking → text-available must feel instant. We measure
  it (see Benchmark) and optimize it across the 10–15 improvement loops.
- **Accuracy.** Real transcription quality, measured as WER against a reference.
- **Pluggable STT.** Four interchangeable backends behind one interface so we
  can benchmark and swap the winner in.

## 3. STT backends (the bench-off)

| Backend           | Type   | Model                     | Needs key            | Streaming |
|-------------------|--------|---------------------------|----------------------|-----------|
| faster-whisper    | local  | configurable (e.g. `small`, `medium`, `large-v3`) | none | chunked |
| Sarvam AI         | cloud  | `saarika` (Sarvam STT)    | `SARVAM_API_KEY`     | batch     |
| Groq              | cloud  | `whisper-large-v3-turbo`  | `GROQ_API_KEY`       | batch (fast) |
| Deepgram          | cloud  | `nova-3`                  | `DEEPGRAM_API_KEY`   | yes (websocket) |

All four implement the same `STTBackend` interface and emit timing metrics so
the benchmark is apples-to-apples.

## 4. Architecture

```
hotkey press ─▶ AudioRecorder (sounddevice, 16kHz mono PCM)
                     │ press again
                     ▼
              STTBackend.transcribe()  ──▶ TranscriptionResult{text, metrics}
                     │
        ┌────────────┴─────────────┐
   capture mode                command mode
        │                           │
   NoteStore.add()           IntentRouter.route(text)
   (SQLite + FTS5)                  │
        │                  ┌────────┴─────────┐
        ▼               show / today      summarize / ask
   notify + log         NoteStore query   Claude API over notes
                                              │
                                          notify + print
```

### Modules (`src/instant_notes/`)
- `cli.py` — entry point: `start` (daemon), `bench`, `notes`, `config`.
- `daemon.py` — wires hotkeys → recorder → STT → router; long-running loop.
- `hotkeys.py` — global hotkeys (`pynput`). Toggle-to-record. **[ASSUMPTION]**
  capture = `Cmd+Shift+Space`, command = `Cmd+Shift+A` (configurable).
- `audio.py` — record to in-memory buffer + optional WAV; resample to 16k mono.
- `stt/base.py` — `STTBackend` ABC + `TranscriptionResult` + `STTMetrics`.
- `stt/{faster_whisper,sarvam,groq,deepgram}_backend.py` — adapters.
- `stt/registry.py` — name → backend factory; reads config/env.
- `storage.py` — SQLite DAO. `notes(id, created_at, text, source_backend,
  duration_ms, latency_ms, audio_path)`, FTS5 virtual table for search.
- `intent.py` — classify spoken command → {show_recent, today, search,
  summarize, ask}; fulfill using `storage` + Claude (`anthropic` SDK).
- `notify.py` — macOS notification (`osascript`) + terminal output; optional
  paste-at-cursor for capture results. **[ASSUMPTION]** paste-at-cursor OFF by
  default (needs Accessibility perms); notes always saved.
- `config.py` — load `config.toml` + `.env`; defaults + validation.
- `bench/harness.py` — run N backends over a sample, collect metrics.
- `bench/metrics.py` — WER (`jiwer`) + latency aggregation; markdown/JSON report.

### Latency instrumentation
Every transcription records: `record_stop_ts`, `request_sent_ts`,
`first_token_ts` (streaming only), `final_ts`, `total_ms`. The benchmark and the
live daemon both populate `STTMetrics` so we optimize against real numbers.

## 5. Config & secrets
- `.env` (gitignored): `SARVAM_API_KEY`, `GROQ_API_KEY`, `DEEPGRAM_API_KEY`,
  `ANTHROPIC_API_KEY`. `.env.example` committed.
- `config.toml`: hotkeys, `default_backend`, whisper model size, paste-at-cursor
  toggle, intent-router model id. `config.example.toml` committed.

## 6. LLM layer
- Routing + summarize/ask via the `anthropic` SDK.
  **[ASSUMPTION]** routing uses `claude-haiku-4-5` (cheap/fast), summarize/ask
  uses `claude-opus-4-8`. Configurable.

## 7. Testing
- `pytest`. Unit: storage CRUD + FTS, intent classification (Claude mocked),
  backend interface conformance (HTTP mocked for cloud, tiny local model gated
  behind a marker). Bench metrics: WER + latency math on fixtures.

## 8. Benchmark deliverable
`instant-notes bench --audio sample.wav --reference ref.txt` runs all configured
backends and emits a table: backend × {total latency, first-token latency, WER,
cost-estimate}. Markdown report saved under `docs/benchmarks/`.

## 9. Stack / tooling
- Python 3.11+, `pyproject.toml`. Deps: `sounddevice`, `numpy`, `pynput`,
  `faster-whisper`, `groq`, `deepgram-sdk`, `httpx` (Sarvam REST), `anthropic`,
  `jiwer`, `python-dotenv`, `rich` (CLI), `pytest`.
- macOS permissions: Microphone + Accessibility (for global hotkeys / paste).

## 10. Open questions (non-blocking — defaults chosen)
- Exact hotkey combos (defaults above).
- Spoken-response TTS — **deferred** (text-only for v1).
- Paste-at-cursor default — OFF for v1.
- Whisper default model size — **[ASSUMPTION]** `small` for speed; bench will
  inform the real choice.

## 11. Workflow (per user)
GitHub branch/commit/push throughout · multiple subagents for parallel module
builds · this design + project `CLAUDE.md` keep it structured · `/pipeline`
phases · post-build: real bench-off on user audio+keys, then 10–15 improvement
loops.
