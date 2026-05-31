# Voice-Claude Bridge

Real-time voice-to-Claude assistant. Speak into your microphone, see live transcription, and get streaming AI responses.

There are three ways to use it:

1. **Universal push-to-talk dictation** — hold a global hotkey *anywhere* in
   Windows, speak, release, and your speech is typed at the cursor: Word,
   Chrome, a terminal, any text field. This is the main mode. See
   [Push-to-talk dictation](#push-to-talk-dictation).
2. **Voice into Claude Code (terminal)** — a variant that types the transcript
   into your open Claude Code session and presses Enter. See
   [Voice → Claude Code](#voice--claude-code-terminal).
3. **Browser UI** — the web page with live transcript + streaming response,
   handy for debugging the STT pipeline.

## Push-to-talk dictation

Speech-to-text anywhere in Windows. Hold the hotkey, talk, release — the text
is typed wherever your cursor is.

```
[hold hotkey]  → record mic
   (you speak)
[release]      → faster-whisper transcribes → text typed at the cursor
```

No browser, no WebSocket, no API key, no VAD guesswork — the key press/release
defines the speech boundaries, which makes transcription accurate.

### Run it

```bash
.venv\Scripts\activate
python scripts/push_to_talk.py
```

Then focus any text field, hold **`ctrl+shift+space`**, speak, and release.

### Configure (`backend/.env`)

```env
PTT_HOTKEY=ctrl+shift+space # any keyboard-library combo
PTT_OUTPUT_MODE=paste       # 'paste' (clipboard+Ctrl+V, reliable) or 'type'
PTT_TYPE_DELAY_S=0.006      # per-char delay for 'type' mode (raise if garbled)
PTT_APPEND_SPACE=true       # add a space after each dictation
WHISPER_MODEL=small         # tiny/base/small/medium for speed vs accuracy
```

Notes:
- Windows-only (uses Win32 + the `keyboard` library). The default
  `ctrl+shift+space` avoids clashing with common Windows shortcuts like Win+H.
- **`paste` mode** delivers the whole transcript atomically (one Ctrl+V), so it
  can't drop characters — recommended for full sentences. It restores your
  previous clipboard contents afterwards.
- **`type` mode** synthesizes per-character keystrokes. It's paced by
  `PTT_TYPE_DELAY_S`; with too small a delay, fast injection can overflow the
  target app and drop characters (e.g. "Hello, what is happening?" →
  "Hello, ning?"). Raise the delay if you see that.
- The laptop **`Fn` key cannot be used** as a hotkey — it's handled inside the
  keyboard's firmware and never sends a scancode to Windows, so no software can
  detect it. Use a `ctrl`/`shift`-based combo instead.
- Some elevated apps only accept synthesized input if this script also runs
  elevated ("Run as administrator").

## Voice → Claude Code (terminal)

```
Microphone → WebRTC VAD → faster-whisper STT → final transcript
                                                     ↓
                              keystroke injection (clipboard paste + Enter)
                                                     ↓
                              your already-open Claude Code terminal
```

No browser, no WebSocket, no Anthropic API key. It captures the mic locally,
transcribes on pause, and pastes the text into your Claude Code terminal.

### Run it

```bash
# 1) Open Claude Code in a terminal as you normally would:
claude

# 2) In a SEPARATE terminal, start the voice runner:
.venv\Scripts\activate
python scripts/voice_to_claude.py
```

Speak, then pause. The transcript prints to the runner console (for debugging)
and is pasted into your Claude Code terminal followed by Enter. Ctrl+C to stop.

### Targeting the right window

By default the runner types into whatever window is focused, so you can just
keep your Claude Code terminal focused. To make it always target that terminal
even if it isn't focused, set these in `backend/.env`:

```env
CC_WINDOW_TITLE=claude      # substring of the terminal's window/tab title
CC_AUTO_FOCUS=true          # find + focus that window before pasting
CC_INJECT_PRESS_ENTER=true  # submit automatically (set false to review first)
```

Notes:
- Injection is Windows-only (uses Win32 keystroke injection + clipboard paste).
  On other platforms the transcript prints but isn't injected.
- Text is delivered by **pasting**, not character-by-character typing, so it's
  fast and won't accidentally trigger Claude Code's slash-command menu.

## Browser UI (debug / API mode)

Real-time browser experience with live transcription and streaming responses.

```
Microphone → [Browser AudioWorklet] → WebSocket → [FastAPI Backend]
                                                       ↓
                                              WebRTC VAD (20ms frames)
                                                       ↓
                                              faster-whisper STT
                                                       ↓
                                              Claude (Anthropic API)
                                                       ↓
                                              Streaming response → UI
```

## Quick Start

### 1. Setup

```bash
cd d:\Projects\voicePrompt
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Configure

```bash
# Copy the example env file
cp backend/.env.example backend/.env

# Edit backend/.env and set your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Warm Up Model (First Time)

```bash
python scripts/warmup_model.py
```

This downloads the Whisper model (~500MB for `small`) so the first request isn't slow.

### 4. Run

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 5. Use

1. Click the **microphone button** to start recording
2. **Speak** into your microphone
3. Watch the **live transcript** update in real time
4. When you pause, the final transcript is sent to **Claude**
5. Claude's response **streams** token by token
6. **Speak again** at any time to interrupt Claude

## Configuration

All settings are in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required for `api` backend) | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | Claude model to use |
| `LLM_BACKEND` | `api` | `api` (direct Messages API) or `claude_code` (native Claude Code CLI) |
| `CLAUDE_CODE_WORKDIR` | (cwd) | Directory Claude Code operates in (its repo context) |
| `CLAUDE_CODE_PERMISSION_MODE` | `default` | `default`, `acceptEdits`, or `bypassPermissions` |
| `CLAUDE_CODE_ALLOWED_TOOLS` | (CLI default) | Comma-separated tools, e.g. `Read,Grep,Edit,Bash` |
| `CLAUDE_CODE_MAX_TURNS` | `12` | Max agent turns per voice request |
| `PTT_HOTKEY` | `ctrl+shift+space` | Global push-to-talk hotkey (hold to record) |
| `PTT_OUTPUT_MODE` | `paste` | `paste` (clipboard+Ctrl+V, reliable) or `type` (per-char) |
| `PTT_TYPE_DELAY_S` | `0.006` | Per-char delay for `type` mode (raise if garbled) |
| `PTT_MIN_RECORD_S` | `0.3` | Ignore clips shorter than this (accidental taps) |
| `PTT_APPEND_SPACE` | `true` | Append a space after each dictation |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8`, `float16`, or `float32` |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `VAD_MODE` | `2` | VAD aggressiveness (0–3) |
| `MAX_SILENCE_MS` | `700` | Silence before finalizing |
| `MIN_SPEECH_MS` | `300` | Minimum speech duration |
| `PARTIAL_UPDATE_MS` | `250` | Partial transcript interval |

## LLM backends

The voice/transcript pipeline is backend-agnostic. Pick where the final
transcript goes with `LLM_BACKEND`:

- **`api`** (default) — sends the transcript to the Anthropic **Messages API**.
  Plain chat completion, no file or tool access. Needs `ANTHROPIC_API_KEY`.

- **`claude_code`** — drives the **native Claude Code CLI** in headless
  streaming mode (`claude -p --output-format stream-json`). This is the full
  agentic experience: Claude Code can read/edit files and run commands inside
  `CLAUDE_CODE_WORKDIR`, and its streamed output is forwarded to the UI just
  like the API backend.

  Setup:
  ```bash
  npm install -g @anthropic-ai/claude-code
  claude            # run once to authenticate (or set ANTHROPIC_API_KEY)
  ```
  Then in `backend/.env`:
  ```env
  LLM_BACKEND=claude_code
  CLAUDE_CODE_WORKDIR=d:\path\to\your\repo
  CLAUDE_CODE_PERMISSION_MODE=acceptEdits   # let it edit files autonomously
  ```

  The transcript is passed to the CLI via **stdin** (not as a shell argument),
  so spoken text with quotes or special characters can't trigger command
  injection. `bypassPermissions` grants full autonomy — use it only in a repo
  you're comfortable letting the agent modify.

## Project Structure

```
voice-claude-bridge/
  backend/
    app/
      main.py              # FastAPI entry point
      config.py             # Settings from env vars
      audio/
        frames.py           # PCM frame parsing
        buffer.py           # Audio ring buffer
        vad.py              # WebRTC VAD wrapper
      stt/
        whisper_engine.py   # faster-whisper integration
        transcript_state.py # Partial/final transcript tracking
      llm/
        claude_client.py    # Anthropic streaming client
        prompts.py          # System/user prompt templates
      api/
        websocket.py        # WebSocket session handler
        health.py           # Health check endpoint
      utils/
        logging.py          # Colored logging
        timing.py           # Latency measurement
    requirements.txt
    .env.example
  frontend/
    index.html              # Single-page UI
  scripts/
    warmup_model.py         # Pre-download Whisper model
    smoke_test.py           # Basic integration test
```

## Troubleshooting

- **No transcript**: Check that audio is 16kHz mono 16-bit PCM. Check browser console for mic errors.
- **Model download slow**: Run `warmup_model.py` first. Or use `WHISPER_MODEL=tiny` for fastest download.
- **Claude not responding**: Verify `ANTHROPIC_API_KEY` is set correctly.
- **GPU not working**: Ensure CUDA 12 + cuDNN 9 are installed for faster-whisper GPU mode.

## License

MIT
