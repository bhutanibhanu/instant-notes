# STT Tuning Summary (loops 1–12)

Sample: 132 s Hinglish (code-switched), romanized reference. Branch:
`feat/stt-bench-tuning`.

## Outcome vs. start

| Metric | Start (main) | After tuning |
|---|---|---|
| Backends working | 2 of 4 | 4 of 4 |
| Output script | Devanagari | romanized Hinglish (Latin) |
| Accuracy (WER) | 0.99 | **0.26–0.30** |
| Perceived stop→text | 79 s (CPU whisper) | **~0.08 s** (streaming, raw) |
| Notes query ("summarize") | dormant (needed Anthropic) | live on Groq |
| Keys required | 4 (incl. Anthropic) | **1 (Groq)** |

## Architecture (production path)
1. **Capture** → audio fed to the STT backend **during recording** in ~8 s
   windows (`_GroqStreamingSession`); each window transcribed in the background.
2. **Stop** → only the final partial window remains → raw transcript in ~80 ms.
3. **Save + show raw immediately** (instant feel).
4. **Background LLM cleanup** (Groq Llama 3.3 70B) fixes spelling/mishears and
   updates the note ~1.5 s later (WER ~0.43 → ~0.30).
5. Silence trimming before STT; FTS-safe search; SQLite storage.

## Backend choice
- **Groq** (default): STT ~0.7 s batch / ~0.08 s streamed; romanized via
  `language=en` + prompt; WER 0.43 raw → 0.30 cleaned. Lowest latency.
- **Sarvam**: STT + transliterate API → WER 0.29 (best); ~3 s short note. Pick
  for max accuracy via `default_backend = "sarvam"`.
- **Deepgram**: fastest raw (~1.8 s) but lossy on Hindi.
- **faster-whisper**: dropped (79 s on CPU).

## Per-loop log
1 fix all 4 backends · 2 romanized Hinglish + Groq default · 3 parallelize Groq
chunks · 4 prompt A/B (kept) · 5 Sarvam transliterate (WER 0.29) · 6 parallelize
Sarvam · 7 language hint (no gain) · 8 silence trim · 9 Groq LLM layer (no
Anthropic) + live query · 10 cleanup-model A/B (kept 70B) · 11 transcribe-while-
recording (787→78 ms) · 12 async cleanup (instant raw + background refine).
