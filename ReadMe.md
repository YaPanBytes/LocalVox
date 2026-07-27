# Voice Assistant (Offline)

A real-time, audio-in/audio-out conversational assistant that runs fully
offline, keeps end-to-end response latency close to 2 seconds, and never
leaves the user with dead air or a generic error if a response is slow.

# Working Demo Video
https://drive.google.com/file/d/1y9Ibj9zWD4PZ7_oZLfqItw3AZhwCe9fk/view?usp=sharing

## Setup

1. Install Python dependencies:
   ```
   pip install faster-whisper ollama sounddevice webrtcvad numpy PyYAML nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```

2. Install [Ollama](https://ollama.com/download) and pull the model:
   ```
   ollama pull llama3.2:1b
   ```

3. Install [Piper](https://github.com/rhasspy/piper/releases) and place the
   executable at `piper/piper.exe` (relative to the project root), or update
   `piper_exe_path` in `config.yaml` to point at wherever you placed it.

4. Download a Piper voice model (e.g. `en_US-lessac-medium`) into
   `models/piper_voices/`.

5. **GPU acceleration for ASR:** faster-whisper needs `cublas64_12.dll` and
   `cudnn64_9.dll` resolvable on PATH. Installing `nvidia-cublas-cu12` and
   `nvidia-cudnn-cu12` via pip (step 1) provides these DLLs, but their `bin`
   folders need to be added to your PATH environment variable — add both:
   ```
   <your-venv-path>\Lib\site-packages\nvidia\cublas\bin
   <your-venv-path>\Lib\site-packages\nvidia\cudnn\bin
   ```
   A permanent PATH edit (not a per-session one) is recommended, since
   session-only PATH changes don't survive a terminal restart.

6. Run:
   ```
   python main.py
   ```
   Speak after "Voice assistant ready." Press `Ctrl+C` to exit.

### Standalone diagnostic scripts

Useful for isolating and debugging each pipeline stage independently:

- `scripts/test_asr.py` — mic capture + VAD + faster-whisper only
- `scripts/test_llm.py` — Ollama streaming + timing only, text input
- `scripts/generate_fillers.py` — one-time generation of fallback filler
  audio clips via Piper

## Architecture

### Pipeline overview

```
Mic input (VAD-gated capture)
        |
        v
faster-whisper (ASR)  ──────►  transcript
        |
        v
Ollama / local LLM (streamed generation)
        |
        v
Piper (TTS, per-sentence streaming)
        |
        v
Speaker playback
```

The system is a plain local Python process (`main.py`) — no web server, no
API layer. It runs a continuous loop: listen → transcribe → respond → speak
→ back to listening.

### State machine

Each conversation turn moves through five states, managed in
`core/orchestrator.py`:

```
listening → transcribing → thinking → speaking → idle
```

- **listening** — mic capture with voice activity detection (VAD) instead of
  a fixed silence timeout, so end-of-speech is detected as soon as the user
  actually stops talking rather than waiting a fixed N seconds.
- **transcribing** — the captured audio is passed to faster-whisper.
- **thinking** — the transcript goes to the LLM. This is also where the
  fallback timer (see below) is running.
- **speaking** — synthesized audio plays, per-sentence (see below).
- **idle** — turn complete, ready to listen again.

### Streaming design (why latency stays low)

The LLM and TTS stages aren't run in the naive "wait for the full response,
then synthesize all of it" sense. Instead:

1. The LLM streams tokens as they're generated (`respond_stream()` in
   `llm/base.py` yields incrementally, not all at once).
2. The orchestrator buffers tokens until it detects a complete sentence
   (via a sentence-boundary regex).
3. As soon as the **first sentence** is complete, it's sent to Piper and
   played — while the LLM is still generating the rest of the response in
   the background.

The user hears the assistant start speaking after the *first sentence* is
ready, not after the *entire response* is ready — this is the single
biggest latency win in the design.

### Fallback design (engagement, not error messages)

Per the requirement that the assistant keep the user engaged rather than
leaving them waiting, `core/fallback.py` implements a timeout-based filler
system:

1. The instant the "thinking" state begins, a background timer starts
   (`fallback_timeout_sec` in `config.yaml`).
2. If the first sentence isn't ready before the timer fires, a
   **pre-generated** filler clip (e.g. "Let me think about that for a
   second") plays immediately — no generation latency, since it's just a
   stored audio file.
3. As soon as the real first sentence is ready, the timer is cancelled. If
   it already fired, the real response plays right after the filler
   finishes.

Filler clips are generated once, offline, via `scripts/generate_fillers.py`
(which calls Piper directly on a fixed list of phrases) and stored as raw
PCM in `assets/fillers/`. Nothing about the fallback path requires network
access or live generation.

### Offline-only, by design

Per the assignment's stated preference for an offline implementation, and
since offline proved fully feasible on the test hardware, this system was
built as offline-only — no online fallback path was implemented.

### Hardware / model sizing decisions

Testing was done on a laptop GPU with 2GB VRAM (NVIDIA MX450) — a
meaningful constraint on model choice, resolved through direct measurement
rather than assumption:

| Model | GPU/CPU split (via `ollama ps`) | Time to first token |
|---|---|---|
| `llama3.2:3b` (Q4) | 17% GPU / 83% CPU | ~9,700ms |
| `llama3.2:1b` (Q4) | 28% GPU / 72% CPU | ~650-700ms |

The 3B model doesn't fit comfortably in 2GB of VRAM alongside its KV cache,
so Ollama silently offloaded most of it to CPU — which caused the
order-of-magnitude latency difference. The 1B model was selected for the
final system based on this measured result.

### Measured end-to-end performance

| Stage | Time |
|---|---|
| Mic capture (VAD-gated) | ~2.5-5s (depends on how long the user speaks) |
| ASR (faster-whisper `base`, GPU) | ~950-1000ms |
| LLM time-to-first-sentence (`llama3.2:1b`) | ~650-700ms |
| **First audio activation (mic-stop to speaking-start)** | **~1.8-2.2s** |

"First audio activation" is measured precisely as: time from end-of-capture
to the first synthesized sentence beginning playback — not total response
generation time, which naturally grows with longer responses and isn't a
meaningful latency metric on its own.

### Known limitations

- Piper is invoked as a subprocess per sentence, which has minor
  process-spawn overhead compared to a persistent in-process TTS engine.
  Acceptable tradeoff for implementation simplicity given the latency
  budget was still met.
- The fallback filler doesn't interrupt itself if the real response becomes
  ready mid-playback of the filler clip — it finishes the filler, then
  plays the real response. Deliberate simplicity tradeoff.
- Conversation history (`core/session.py`) is kept in memory only and
  resets on restart — no persistence across sessions.

## Models used

| Component | Model | Notes |
|---|---|---|
| ASR | faster-whisper `base` | GPU (float16), CUDA |
| LLM | `llama3.2:1b` (Q4, via Ollama) | Selected over `3b` after measuring GPU offload — see above |
| TTS | Piper — `en_US-lessac-medium` | ONNX voice model |
