## Blockers
none

## Verdict
SHIP

## Reasoning
`NoteStore` now uses `check_same_thread=False` with an `RLock` around connection access, and Groq streaming `finish()` raises when every window fails so daemon batch fallback is reached. `_GroqStreamingSession.close()` shuts down the pool, and daemon shutdown closes any active session plus any backend with a callable `close()`.
