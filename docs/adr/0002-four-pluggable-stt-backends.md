# ADR 0002 — Four pluggable STT backends behind one interface

- Status: accepted
- Date: 2026-06-08

## Context
The headline requirement is Wispr-Flow-class latency + accuracy. No single STT
engine is obviously best across latency, accuracy, cost, language coverage, and
offline capability. The user wants to benchmark candidates on their own voice
before committing.

## Decision
Define one `STTBackend` interface (`stt/base.py`) and implement four adapters:

- **faster-whisper** (local) — zero network latency, free, offline; CPU int8 by
  default (CTranslate2 has no Apple-Metal backend, so CPU is the only sane mac
  target).
- **Sarvam AI** (cloud) — strong on Indian English/accents.
- **Groq** whisper-large-v3-turbo (cloud) — extremely fast inference; batch.
- **Deepgram** nova-3 (cloud) — true streaming via websocket (see ADR 0001).

A `registry.build_backend(name, cfg)` factory constructs by name with lazy
imports, so a missing optional SDK only errors if that backend is requested.
All backends emit `STTMetrics` (total / first-token / realtime-factor) so the
`bench` harness compares them apples-to-apples on the user's audio.

## Consequences
- Swapping the production default is a one-line config change after the bench-off.
- Each cloud SDK is an optional extra; the package imports and the local backend
  runs with no keys installed.
- Cost/latency/accuracy tradeoffs are deferred to data (the bench-off) rather
  than guessed.
