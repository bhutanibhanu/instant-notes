## Blockers
- `src/instant_notes/daemon.py:41` / `src/instant_notes/hotkeys.py:21` / `src/instant_notes/daemon.py:144`: `NoteStore` is created on the daemon thread but hotkey callbacks run on `pynput`’s listener thread, so `self.store.add_note(...)`, command queries, and assistant reads can hit SQLite’s default `check_same_thread` error in real daemon use. Tests call `_toggle_*` directly and miss this.
- `src/instant_notes/stt/groq_backend.py:203`: streaming worker exceptions are swallowed and converted to empty text, so `_transcribe()` never sees a failed `session.finish()` and never falls back to batch. A transient Groq/network failure can silently save a partial transcript or report “Heard nothing.”
- `src/instant_notes/daemon.py:213`: shutdown while a Groq streaming recording is active only stops the recorder; it never finishes/cancels the `_GroqStreamingSession`, so the executor created at `src/instant_notes/stt/groq_backend.py:160` can leak non-daemon worker threads and in-flight network calls.

## Non-blocking issues
- `src/instant_notes/stt/sarvam_backend.py:56`: the pooled `httpx.Client` has a `close()` method, but daemon shutdown never calls backend cleanup; Groq/LLM cached clients also have no close path.
- `src/instant_notes/llm.py:16`: global cached LLM clients are unsynchronized and shared between background refinement and command handling; this can race client construction and assumes SDK client thread-safety without a guard.
- `src/instant_notes/stt/groq_backend.py:74`: Groq client lazy construction is also unsynchronized while batch chunking and streaming workers can call `_groq()` concurrently.
- `src/instant_notes/refine.py:26`: cleanup silently suppresses all LLM errors through `llm.chat`, which is okay for UX but not honest operationally; there is no log/metric to tell whether async cleanup is actually working.
- `tests/test_cloud_backends.py:36`: the Groq streaming test comment says 12s windows, but the implementation cap is 14s; minor, but it signals the latency claim is not being tested precisely.

## Suggested tests
- Hotkey-thread integration test that invokes `_toggle_capture` from a different thread than daemon construction and verifies SQLite writes/reads do not raise.
- Groq streaming worker failure test where `_transcribe_window` raises and daemon proves it falls back to batch instead of saving empty/partial text.
- Daemon shutdown test with an active streaming session that asserts executor/session cleanup is called.
- Concurrent LLM cleanup plus notes-query test to exercise shared cached clients.
- Sarvam/Groq parallel chunk tests that simulate one failed chunk, timeout, and malformed response.
- End-to-end async refine test that verifies raw note is saved immediately, refined text updates FTS, and cleanup failure leaves raw text unchanged with observable logging.
- I could not run the suite here because `pytest` is not installed in the environment.

## Verdict
NO_SHIP

## Reasoning
The PR broadly matches the tuning-summary architecture, but the production daemon still has a SQLite cross-thread bug and the new streaming path can silently drop transcription failures instead of falling back. The tests cover happy-path wiring and chunk counts, but they do not exercise the concurrency, cleanup, shutdown, or network-failure paths that are most likely to break this feature in real use.
