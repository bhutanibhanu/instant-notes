# CLAUDE.md — Instant Notes

Project-level instructions for Claude Code. These were given directly by the
user and override defaults. Read this before working in this repo.

## What we're building
A macOS background-daemon voice notes app. Global hotkey starts/stops recording;
speech is transcribed (low-latency, high-accuracy — Wispr-Flow class) and saved
as a timestamped note. A second hotkey + voice command lets the user query their
notes ("show notes", "summarize", "what did I note today") answered via Claude.

Full design: `docs/features/voice-notes/design.md`.

## How the user wants me to work (non-negotiable)
1. **Paperclip — agent orchestration** (https://github.com/paperclipai/paperclip).
   Open-source control plane that orchestrates teams of AI agents (Claude Code,
   Codex, etc.) above the individual-tool layer: persistent sessions, per-agent
   token budgets, approval workflows, task tickets w/ dependencies, cron/webhook
   scheduling. The user wants this build run as coordinated, GitHub-tracked
   agents. NOTE: Paperclip is a separate Node/React app (`npx paperclipai
   onboard --yes`) run OUTSIDE a Claude Code session — it drives sessions, it
   isn't callable from within one. In-session, the equivalent is subagents
   (line 3). Stand Paperclip up only if the user wants external orchestration.
2. **GitHub workflow throughout** — feature branch, frequent small commits, push
   regularly. Current branch: `feat/voice-notes`. Repo:
   `github.com/bhutanibhanu/instant-notes`.
3. **Multiple subagents** — parallelize independent module builds (STT adapters,
   audio, storage, intent, bench) across subagents. Architect shared interfaces
   first, then fan out. This is the in-session realization of the Paperclip
   "team of agents" model.
4. **Claude workflows / `/pipeline` skill** — drive the lifecycle with Claude's
   agentic workflow: scope → build → handoff → Codex QA → ship. Don't skip the
   QA checkpoint.
5. **Extremely structured** — design doc, project docs, clear module boundaries,
   tests, benchmark reports. No ad-hoc sprawl.
6. **Model** — run on Opus 4.8 with 1M context; use high reasoning effort and
   subagents for heavy work.

## The 4 STT backends (must stay pluggable & benchmarkable)
faster-whisper (local), Sarvam AI, Groq (whisper-large-v3-turbo), Deepgram
nova-3. One `STTBackend` interface; all emit latency + accuracy metrics.

## API keys needed (user provides at bench time)
`SARVAM_API_KEY`, `GROQ_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`.
Local faster-whisper needs none. Keys live in `.env` (gitignored).

## After the build
1. User provides a voice sample + the API keys above.
2. Run `instant-notes bench` to compare all 4 backends on latency + WER.
3. Then 10–15 improvement loops to push latency down and accuracy up — make it
   the best possible.

## Conventions
- Python 3.11+, `pyproject.toml`, `src/instant_notes/` layout.
- `pytest` for tests. Mock cloud APIs in unit tests.
- Never commit secrets. `.env.example` / `config.example.toml` only.
