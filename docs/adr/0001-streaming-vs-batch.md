# ADR 0001 — Streaming vs batch transcription

- Status: accepted
- Date: 2026-06-08

## Context
The headline requirement is Wispr-Flow-class latency: the final transcript
should be ready the *instant* the user releases the hotkey. The original design
was purely **batch** — record the whole clip, then send it / decode it and wait.
That serializes recording time + transcription time, so a 20 s note pays a
multi-second tail after the user stops.

We want transcription to overlap recording: ship/decode audio *while the user
is still speaking* so that on stop only the trailing fragment remains.

The four backends differ sharply in how they can stream:

- **Deepgram nova-3** exposes a true incremental **websocket** API with interim
  and final results — ideal for live streaming.
- **faster-whisper** (CTranslate2) has **no incremental decoder**; it decodes a
  whole array per call.
- **Groq / Sarvam** are batch HTTP endpoints today.

So streaming has to be *optional and additive* — not every backend can do it,
and the batch path must keep working unchanged for the bench-off.

## Decision
Add a small streaming layer alongside (not replacing) the batch contract:

- `StreamingSession` ABC in `stt/base.py` with non-blocking `feed(chunk)`,
  `finish() -> TranscriptionResult`, and best-effort `latest_partial()`.
- `STTBackend.supports_streaming()` (default `False`) and `open_stream(rate)`
  (default `NotImplementedError`). Batch `transcribe()` is untouched.
- `AudioRecorder.start(on_chunk=...)` delivers each captured block live (mono
  float32) on the PortAudio thread; `feed` only enqueues, so the audio thread
  never blocks. `stop()` still returns the full `AudioData` so we keep a saved
  copy and the batch path works.
- The daemon prefers streaming when the active backend supports it, opening a
  session and wiring `on_chunk=session.feed`; on stop it calls both
  `recorder.stop()` (for the saved clip + duration) and `session.finish()` (for
  the transcript). Any failure opening or finishing the stream logs and falls
  back to batch `transcribe()`.

Backend specifics:

- **Deepgram**: real websocket streaming (`listen.websocket.v("1")` +
  `LiveOptions(..., interim_results=True)`). `feed` converts float32 → int16 LE
  bytes and enqueues; a worker thread `send()`s them. A transcript handler
  accumulates FINAL segments and tracks the latest interim. `finish()`
  finalizes, waits briefly (≤3 s) for trailing finals, and returns the joined
  text. The `DeepgramClient` is cached and reused for batch too.
- **faster-whisper**: **windowed** streaming. There is no incremental Whisper
  decoder, so the session re-transcribes the *whole accumulated buffer* every
  ~3 s on a worker thread to refresh the partial, and runs one final pass on
  `finish()`.

## Consequences
- Low latency where it matters most: Deepgram returns the transcript almost
  immediately on stop because nearly all audio was already decoded in flight.
- **O(n²) re-transcription for faster-whisper**: re-decoding the full buffer
  every window is quadratic in note length. This is an accepted tradeoff for
  *short* voice notes (the target use case); a true incremental / chunked
  decoder (e.g. fixed-window with context carry-over) is future work. The final
  pass still re-decodes everything, so accuracy matches batch.
- The batch path is fully preserved, so the bench-off (`instant-notes bench`)
  keeps comparing all four backends apples-to-apples.
- `feed` is strictly non-blocking (enqueue + return), protecting real-time audio
  capture from network or decoder stalls.
