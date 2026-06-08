# Instant Notes

Hotkey-driven voice notes for macOS. Press a hotkey to start recording, press
again to stop — your speech is transcribed and saved as a timestamped note.
Press a second hotkey and speak a query ("show my notes", "summarize my notes",
"what did I note today") to get answers from your notes via Claude.

Built for **low latency and high accuracy** (Wispr-Flow class), with **four
pluggable STT backends** you can benchmark head-to-head.

## STT backends

| Backend           | Type  | Needs key          |
|-------------------|-------|--------------------|
| faster-whisper    | local | none               |
| Sarvam AI         | cloud | `SARVAM_API_KEY`   |
| Groq (whisper-v3-turbo) | cloud | `GROQ_API_KEY` |
| Deepgram nova-3   | cloud | `DEEPGRAM_API_KEY` |

The query layer uses the Claude API (`ANTHROPIC_API_KEY`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"        # or ".[dev]" for local-whisper only
cp .env.example .env               # then fill in your API keys
cp config.example.toml config.toml # optional: tweak hotkeys / model
```

macOS will prompt for **Microphone** and **Accessibility** permissions on first
run (Accessibility is needed for global hotkeys / paste-at-cursor).

## Usage

```bash
instant-notes start                # run the daemon (listens for hotkeys)
instant-notes notes                # list recent notes
instant-notes notes --search foo   # full-text search
instant-notes bench --audio sample.wav --reference ref.txt   # bench-off
```

## Design

See [`docs/features/voice-notes/design.md`](docs/features/voice-notes/design.md)
and [`CLAUDE.md`](CLAUDE.md).
