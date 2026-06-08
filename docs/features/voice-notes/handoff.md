# Feature Handoff: voice-notes

## Goal
A macOS background daemon: hotkey to dictate a timestamped voice note (low-latency,
high-accuracy STT), and a second hotkey to speak queries answered from the notes
("show / today / search / summarize / ask") via Claude.

## Files changed
40 files, +3329 / -5 vs `main`. Highlights:
- `src/instant_notes/stt/` — `base.py` (STTBackend + StreamingSession interfaces),
  4 backends (`faster_whisper`, `sarvam`, `groq`, `deepgram`), `registry.py`.
- `src/instant_notes/{audio,storage,config,intent,notify,hotkeys,daemon,cli}.py`.
- `src/instant_notes/bench/` — WER + latency harness + report.
- `tests/` — 6 test modules (39 passing, 1 gated skip).
- `docs/adr/0001–0003`, `docs/features/voice-notes/design.md`, CI workflow.

## How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"        # all = faster-whisper + groq + deepgram SDKs
cp .env.example .env               # fill SARVAM/GROQ/DEEPGRAM/ANTHROPIC keys
instant-notes start                # daemon: Cmd+Shift+Space dictate, Cmd+Shift+A query
instant-notes notes [--search Q] [--today]
instant-notes bench --audio sample.wav --reference ref.txt
```
Tests/quality: `pytest -q`, `ruff check src tests`, `mypy`.

## Expected behavior
- Capture hotkey: toggle record → transcribe → save note (+ optional paste-at-cursor).
- Command hotkey: toggle record → transcribe → intent-route → notify + print answer.
- If the backend supports streaming (Deepgram live, faster-whisper windowed),
  transcription runs *during* recording; otherwise batch on stop. Streaming
  failures fall back to batch.
- Retrieval intents (show/today/search) work offline with no API key; summarize/ask
  use Claude and degrade gracefully without `ANTHROPIC_API_KEY`.

## Test plan
- **Automated (hermetic, mocked):** storage+FTS, intent routing (fake Claude),
  cloud backends (mocked HTTP/SDK), bench WER/latency + error handling, CLI
  (version/notes/bench-args), streaming (fake session + daemon path). 39 pass.
- **Needs manual / real keys (not yet done):**
  - End-to-end with a real mic + Microphone/Accessibility permissions.
  - Real API calls per backend (no key in build env) — esp. Sarvam request shape
    and Deepgram **live websocket event signatures**.
  - The bench-off on a real voice sample (latency + WER comparison).

## Known risks
- **Deepgram streaming event handler** (`_DeepgramStreamingSession`) is coded to
  the v3 SDK contract but unexercised against a live socket — the `is_final` /
  `UtteranceEnd` shapes and handler signature could differ. First place to look
  if streaming yields empty text.
- **Sarvam** request shape (multipart + `language_code`) is unverified against the
  live API; may need field tweaks on first real call.
- **faster-whisper windowed streaming** re-transcribes the whole buffer every ~3s
  (O(n²)); fine for short notes, revisit for long dictation (see ADR 0001).
- Global hotkeys/paste need macOS Accessibility; mic needs Microphone permission.

## Open questions
- Final default backend — decide from the bench-off data.
- Whisper model size default (`small`) — tune for the latency/accuracy target.
- Spoken (TTS) responses — deferred; text-only for v1.
