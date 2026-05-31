# implementation.md — Voice-to-Claude Assistant

## 1) Goal

Build a separate project that captures microphone speech, transcribes it in near real time with **faster-whisper**, detects speech boundaries with **VAD**, sends the final transcript to **Claude**, and streams Claude’s response back to the UI. Use Anthropic’s **Messages API** with streaming, because this is direct model prompting with your own control loop, not a long-running managed-agent workflow. Anthropic’s official SDKs support streaming, retries, and error handling, and the current Claude docs position **Haiku** as the fastest model, **Sonnet** as the coding/workflow model, and **Opus** as the most capable model. ([Claude API Docs][1])

## 2) Recommended stack

Use:

* **Backend:** Python 3.10+.
* **STT:** `faster-whisper`.
* **VAD:** `webrtcvad` for a simple streaming MVP, or Silero VAD if you want a stronger VAD later.
* **LLM:** Anthropic Python SDK (`anthropic`).
* **Realtime transport:** WebSocket.
* **API layer:** FastAPI + Uvicorn.
* **Optional frontend:** React or a minimal HTML page.

This is a good fit because faster-whisper is a CTranslate2-based reimplementation of Whisper, is reported to be faster and lighter than `openai/whisper`, and it can auto-download model weights from Hugging Face when you load by size name. Anthropic’s Python SDK is the official way to call Claude from Python, and Claude’s streaming API exposes token/text streaming directly. ([GitHub][2])

## 3) Non-goals

Do not build:

* speaker diarization,
* multilingual translation,
* wake-word detection,
* long-term memory,
* full agentic task execution.

This project should be a **voice command bridge** first. Keep it boring and reliable. Boring wins. Chatty robotics demos lose.

## 4) Target user flow

1. User speaks into the microphone.
2. VAD detects speech start.
3. Audio is buffered in short frames.
4. faster-whisper emits partial transcript updates.
5. When the user pauses, the final transcript is committed.
6. Final transcript is sent to Claude.
7. Claude streams its reply token by token.
8. UI shows both the transcript and Claude output live.

## 5) Repository structure

```text
voice-claude-bridge/
  backend/
    app/
      main.py
      config.py
      audio/
        capture.py
        frames.py
        vad.py
        buffer.py
      stt/
        whisper_engine.py
        transcript_state.py
      llm/
        claude_client.py
        prompts.py
      api/
        websocket.py
        health.py
      utils/
        logging.py
        timing.py
    requirements.txt
    .env.example
  frontend/
    src/
    package.json
  scripts/
    warmup_model.py
    smoke_test.py
  docs/
    implementation.md
    architecture.md
  README.md
```

## 6) Environment setup

### 6.1 Python

Use Python 3.10 or newer for the project runtime. faster-whisper itself requires Python 3.9 or greater, and the Claude SDK is best treated as a modern Python dependency stack, so there is no reason to run this on an old interpreter. ([GitHub][2])

### 6.2 Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 6.3 Install backend dependencies

```bash
pip install --upgrade pip
pip install fastapi uvicorn[standard] websockets python-dotenv pydantic numpy sounddevice webrtcvad anthropic faster-whisper
```

If you use GPU inference, also install the appropriate CUDA/cuDNN stack required by CTranslate2. The faster-whisper project notes that current CTranslate2 versions support CUDA 12 and cuDNN 9, and it documents downgrade workarounds for older CUDA/cuDNN combinations. It also notes that faster-whisper does not require a separate system FFmpeg install because PyAV bundles the FFmpeg libraries. ([GitHub][2])

### 6.4 Anthropic API key

Set the API key as an environment variable named `ANTHROPIC_API_KEY`; the SDK reads it automatically. Use the official Anthropic Console to create the key. ([Claude API Docs][3])

```bash
# Windows PowerShell
setx ANTHROPIC_API_KEY "your_key_here"

# macOS/Linux
export ANTHROPIC_API_KEY="your_key_here"
```

### 6.5 Optional `.env.example`

```env
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-haiku-4-5
WHISPER_MODEL=distil-large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
SAMPLE_RATE=16000
VAD_MODE=2
MAX_SILENCE_MS=700
MIN_SPEECH_MS=300
PARTIAL_UPDATE_MS=250
```

## 7) Model choice and download plan

### 7.1 Claude model

Make the Claude model configurable. For low-latency command handling, default to the fastest model available to your account, and allow switching to a stronger model for harder tasks. Anthropic’s current docs list **Claude Haiku 4.5** as the fastest model, **Claude Sonnet 4.6** as the coding/agent workflow model, and **Claude Opus 4.8** as the most capable model. ([Claude API Docs][1])

### 7.2 Whisper model

Use `faster-whisper` with a configurable model name. The repo shows examples with `"large-v3"` and `"distil-large-v3"`, and it states that loading by size name automatically downloads the corresponding CTranslate2 model from the Hugging Face Hub. It also supports loading a local converted model directory, which is the right path for offline or prepacked deployments. ([GitHub][2])

Recommended default:

* `distil-large-v3` for high quality with lower latency on capable hardware.
* `small` or `medium` for CPU-first MVPs.
* `large-v3` only if you can afford the latency/memory hit.

### 7.3 Pre-download / warmup script

Create `scripts/warmup_model.py` that instantiates the model once at startup so the first real user utterance does not pay the download penalty.

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "distil-large-v3",
    device="cpu",
    compute_type="int8",
)
print("Model loaded")
```

If you need offline packaging, convert a Whisper model to CTranslate2 format and load it from a local directory. faster-whisper’s README documents `ct2-transformers-converter` and direct local-directory loading. ([GitHub][2])

## 8) Audio pipeline design

### 8.1 Audio capture

Capture microphone audio at:

* **16 kHz**
* **mono**
* **16-bit PCM**

WebRTC VAD expects 16-bit mono PCM and accepts 8 kHz, 16 kHz, 32 kHz, or 48 kHz audio with frame sizes of 10, 20, or 30 ms. That makes it a strong fit for frame-based speech boundary detection. ([GitHub][4])

### 8.2 VAD strategy

Use `webrtcvad` first because it is simple and frame-oriented. Start with mode `2` or `3`:

* `0` = least aggressive
* `3` = most aggressive

Keep a small ring buffer of pre-roll audio so the start of speech is not clipped.

Recommended VAD state machine:

* `IDLE`
* `LISTENING`
* `SPEAKING`
* `COOLDOWN`
* `FINALIZING`

### 8.3 Chunking rules

Use short frames:

* 20 ms frames for VAD
* 250 ms partial transcription batches
* 500–800 ms silence timeout to end an utterance

A good first implementation is:

* stream frames into a rolling buffer,
* once speech starts, keep buffering,
* every `PARTIAL_UPDATE_MS`, run transcription on the buffered speech,
* on silence longer than `MAX_SILENCE_MS`, finalize the utterance.

### 8.4 Why not just rely on Whisper alone

faster-whisper does include a VAD filter option and integrates Silero VAD for silence removal, but for a low-latency interactive voice bridge you want explicit control over turn boundaries, interruption, and partial transcript cadence. Use external VAD for the UI/turn logic and Whisper for transcription. ([GitHub][2])

## 9) Backend API design

Use a single WebSocket for the live session.

### 9.1 Client → server messages

```json
{ "type": "audio_frame", "pcm": "<base64>", "seq": 123 }
{ "type": "reset" }
{ "type": "stop" }
```

### 9.2 Server → client messages

```json
{ "type": "vad_state", "state": "speaking" }
{ "type": "partial_transcript", "text": "open the..." }
{ "type": "final_transcript", "text": "open the GitHub repo and review it" }
{ "type": "claude_delta", "text": "Sure — here's the plan..." }
{ "type": "claude_done" }
{ "type": "error", "message": "..." }
```

### 9.3 Session state

Maintain per-connection state:

* audio ring buffer,
* VAD state,
* transcript accumulator,
* Claude response stream state,
* cancellation token for interruption.

## 10) STT module design

### 10.1 Responsibilities

`stt/whisper_engine.py` should:

* own the Whisper model,
* transcribe buffered speech segments,
* emit partial and final text,
* normalize output text,
* keep latency low.

### 10.2 Transcription settings

Use:

* `beam_size=1` for lowest latency in live mode,
* `condition_on_previous_text=False` for segment independence in streaming mode,
* `word_timestamps=False` initially,
* `vad_filter=False` if you are already doing external VAD.

faster-whisper’s README shows `beam_size=5` in examples, but for a live voice assistant you should optimize for responsiveness first and quality second. That choice is an engineering tradeoff, not a library requirement. ([GitHub][2])

### 10.3 Partial transcript strategy

Do not try to re-transcribe the entire conversation every time. Keep an active utterance buffer and only re-run transcription over the current speech chunk. Use a simple diff algorithm to replace only the changed tail of the partial transcript.

### 10.4 Final transcript strategy

Finalize an utterance when:

* speech has ended,
* silence exceeds threshold,
* user presses a stop key,
* or a maximum utterance duration is reached.

Then freeze the final transcript and hand it to the Claude client.

## 11) Claude client design

### 11.1 Use the Messages API

Use the Messages API because this project needs direct prompting and streamed responses. Anthropic’s docs describe Messages API as direct model prompting, while managed agents are intended for long-running asynchronous workflows. The Python SDK also supports streaming, both sync and async. ([Claude API Docs][1])

### 11.2 System prompt

Create a compact system prompt that makes Claude behave like a task assistant.

Example:

```text
You are a voice-controlled coding and task assistant.
Be concise.
If the user asks for multi-step work, give the plan first, then execute.
If the request is ambiguous, ask one clarifying question only when necessary.
Prefer actionable output.
```

### 11.3 User prompt format

Wrap the final transcript in a stable format:

```text
User voice transcript:
<final transcript here>

Task:
Respond to the user’s request.
```

### 11.4 Streaming response

Use Anthropic’s streaming interface and forward text deltas to the client as they arrive. The official docs show `client.messages.stream(...)` and iteration over `stream.text_stream` for token/text streaming. ([Claude API Docs][5])

## 12) Backend implementation order

### Phase 1 — project skeleton

Create:

* FastAPI app,
* WebSocket endpoint,
* config loader,
* logging,
* health endpoint.

Acceptance:

* server boots,
* `/health` returns OK,
* WebSocket accepts connection.

### Phase 2 — microphone ingest

Create:

* audio frame parser,
* PCM validation,
* sample-rate normalization,
* ring buffer.

Acceptance:

* client can send audio frames,
* server stores them without dropping.

### Phase 3 — VAD

Create:

* `webrtcvad` wrapper,
* speech-start detection,
* speech-end detection,
* pre-roll and hangover handling.

Acceptance:

* speaking triggers `speaking`,
* silence ends the utterance.

### Phase 4 — partial STT

Create:

* Whisper model loader,
* buffer-to-transcription function,
* partial transcript emitter.

Acceptance:

* partial text appears while user is speaking.

### Phase 5 — final STT commit

Create:

* finalization logic,
* transcript freeze,
* dispatch to Claude.

Acceptance:

* final transcript is stable and complete.

### Phase 6 — Claude streaming

Create:

* Anthropic client wrapper,
* streaming response generator,
* event forwarding to frontend.

Acceptance:

* Claude text streams live, not all at once.

### Phase 7 — interruption

Create:

* cancel current Claude stream when the user starts talking again,
* reset audio/transcript state cleanly.

Acceptance:

* user can interrupt the assistant mid-answer.

### Phase 8 — polish

Add:

* push-to-talk toggle,
* wake button,
* clipboard copy,
* transcript history,
* optional TTS.

## 13) Suggested backend code skeleton

### `app/config.py`

```python
from pydantic import BaseModel
import os

class Settings(BaseModel):
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    whisper_model: str = os.getenv("WHISPER_MODEL", "distil-large-v3")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    vad_mode: int = int(os.getenv("VAD_MODE", "2"))
    max_silence_ms: int = int(os.getenv("MAX_SILENCE_MS", "700"))
    min_speech_ms: int = int(os.getenv("MIN_SPEECH_MS", "300"))
    partial_update_ms: int = int(os.getenv("PARTIAL_UPDATE_MS", "250"))
```

### `app/llm/claude_client.py`

```python
from anthropic import Anthropic

class ClaudeClient:
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def stream_answer(self, user_text: str, system_prompt: str):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text
```

### `app/stt/whisper_engine.py`

```python
from faster_whisper import WhisperModel

class WhisperEngine:
    def __init__(self, model_name: str, device: str, compute_type: str):
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe_audio(self, audio_path_or_buffer):
        segments, info = self.model.transcribe(
            audio_path_or_buffer,
            beam_size=1,
            condition_on_previous_text=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        return text, info
```

### `app/audio/vad.py`

```python
import webrtcvad

class VADDetector:
    def __init__(self, mode: int = 2):
        self.vad = webrtcvad.Vad(mode)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return self.vad.is_speech(frame, sample_rate)
```

### `app/api/websocket.py`

```python
from fastapi import WebSocket, WebSocketDisconnect

async def voice_session(websocket: WebSocket, session_manager):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            # route audio_frame / reset / stop
    except WebSocketDisconnect:
        await session_manager.close_session(websocket)
```

## 14) Frontend requirements

The frontend should show:

* microphone status,
* live partial transcript,
* final transcript,
* Claude answer stream,
* reconnect/error state,
* start/stop button.

Keep the UI dead simple. The whole point is voice input, not a fancy dashboard cosplay.

## 15) Build notes for Copilot

Tell Copilot to build in this order:

1. **Config and environment loading**
2. **FastAPI app with health check**
3. **WebSocket session scaffolding**
4. **Audio frame normalization**
5. **WebRTC VAD turn detection**
6. **faster-whisper integration**
7. **Partial transcript diffing**
8. **Anthropic streaming integration**
9. **Frontend live rendering**
10. **Cancellation/interruption**
11. **Logging and tests**
12. **Packaging and docs**

## 16) Testing plan

### Unit tests

Test:

* frame parsing,
* VAD decisions,
* turn start/stop logic,
* transcript diffing,
* Claude event forwarding,
* state reset on interruption.

### Integration tests

Test:

* local microphone to transcript,
* final transcript to Claude,
* Claude streaming to frontend,
* interruption during active answer.

### Smoke tests

Use a fixed WAV file and verify:

* utterance segmentation,
* transcript text is produced,
* Claude stream starts,
* connection survives a reset.

## 17) Performance targets

Set these as internal targets:

* speech start detection: fast enough to feel immediate,
* partial transcript refresh: around every 200–300 ms,
* silence-to-finalization: under 800 ms,
* Claude first token after final transcript: as low as possible.

Do not optimize for perfect accuracy first. Optimize for not feeling broken.

## 18) Deployment notes

### Local

* backend on `localhost:8000`
* frontend on `localhost:3000` or Vite default
* WebSocket between them

### Docker

Create separate containers for:

* backend,
* frontend,
* optional model cache volume.

For faster-whisper, pre-warm the model during container startup so the first user is not punished for your cold start.

## 19) Troubleshooting

### No transcript

* Check sample rate is 16 kHz.
* Check audio is mono and 16-bit PCM.
* Check VAD frame size is 10/20/30 ms for WebRTC VAD. ([GitHub][4])

### Model download is slow

* Pre-download the Whisper model on image build or startup.
* Use a smaller Whisper model for MVP.
* Use a local converted CTranslate2 model directory. faster-whisper supports both auto-download from Hugging Face and local directory loading. ([GitHub][2])

### GPU not working

* Verify CUDA 12 + cuDNN 9 compatibility with current CTranslate2.
* Confirm the NVIDIA libraries are installed and visible to the runtime. ([GitHub][2])

### Claude response not streaming

* Make sure you are using the Anthropic SDK stream interface.
* Verify `ANTHROPIC_API_KEY` is set.
* Confirm the selected model exists for your account. ([Claude API Docs][5])

## 20) Final build rule

Ship the smallest version that works:

* one microphone,
* one websocket,
* one VAD,
* one Whisper model,
* one Claude stream.

Everything else is garnish.

[1]: https://docs.anthropic.com/en/docs/intro-to-claude "Intro to Claude - Claude API Docs"
[2]: https://github.com/SYSTRAN/faster-whisper "GitHub - SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2 · GitHub"
[3]: https://docs.anthropic.com/en/docs/get-started "Get started with Claude - Claude API Docs"
[4]: https://github.com/wiseman/py-webrtcvad "GitHub - wiseman/py-webrtcvad: Python interface to the WebRTC Voice Activity Detector · GitHub"
[5]: https://docs.anthropic.com/en/api/messages-streaming "Streaming messages - Claude API Docs"
