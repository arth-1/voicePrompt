"""
WebSocket session handler for the voice-to-Claude pipeline.

Manages per-connection state:
- Audio buffer
- VAD detector
- Transcript state
- Claude response streaming
- Cancellation for interruption

Message protocol:
    Client → Server:
        { "type": "audio_frame", "pcm": "<base64>", "seq": 123 }
        { "type": "reset" }
        { "type": "stop" }

    Server → Client:
        { "type": "vad_state", "state": "speaking" }
        { "type": "partial_transcript", "text": "..." }
        { "type": "final_transcript", "text": "..." }
        { "type": "claude_delta", "text": "..." }
        { "type": "claude_done" }
        { "type": "error", "message": "..." }
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from ..audio.buffer import AudioBuffer
from ..audio.frames import AudioFrameError, parse_audio_frame
from ..audio.vad import VADDetector, VADState
from ..config import settings
from ..llm.base import LLMClient
from ..stt.transcript_state import TranscriptState
from ..stt.whisper_engine import WhisperEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Thread pool for running Whisper transcription (CPU-bound)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")


class VoiceSession:
    """
    Per-connection voice session holding all pipeline state.

    Args:
        websocket: The FastAPI WebSocket connection.
        whisper: Shared Whisper engine instance.
        claude: Shared Claude client instance.
    """

    def __init__(
        self,
        websocket: WebSocket,
        whisper: WhisperEngine,
        claude: LLMClient,
    ):
        self.ws = websocket
        self.whisper = whisper
        self.claude = claude
        self._closed = False

        # Audio pipeline
        frame_duration_ms = 20
        self.audio_buffer = AudioBuffer(
            sample_rate=settings.sample_rate,
            frame_duration_ms=frame_duration_ms,
        )
        self.vad = VADDetector(
            mode=settings.vad_mode,
            sample_rate=settings.sample_rate,
            frame_duration_ms=frame_duration_ms,
            max_silence_ms=settings.max_silence_ms,
            min_speech_ms=settings.min_speech_ms,
        )
        self.transcript = TranscriptState()

        # Claude streaming control
        self._cancel_event = asyncio.Event()
        self._claude_task: Optional[asyncio.Task] = None
        self._is_claude_streaming = False

        # Partial transcript timing — throttle to avoid overwhelming CPU
        self._last_partial_time: float = 0.0
        self._partial_interval_s = settings.partial_update_ms / 1000.0
        self._partial_running = False  # Prevent overlapping partial transcriptions
        self._utterance_id = 0  # Bumped each new utterance; scopes partials

        # Frame size for VAD (20ms at 16kHz = 640 bytes)
        self._vad_frame_bytes = int(settings.sample_rate * frame_duration_ms / 1000) * 2

        # Accumulator for incoming audio that may not align to frame boundaries
        self._frame_remainder = b""

        # Track total frames for debugging
        self._total_frames = 0
        self._speech_frames = 0

    async def send(self, msg: dict) -> None:
        """Send a JSON message to the client."""
        if self._closed:
            return
        try:
            await self.ws.send_json(msg)
        except Exception:
            self._closed = True

    async def handle_audio_frame(self, pcm_b64: str) -> None:
        """
        Process an incoming audio frame through the full pipeline:
        decode → split into VAD frames → VAD → buffer → partial STT.

        Args:
            pcm_b64: Base64-encoded 16-bit PCM audio.
        """
        try:
            pcm_bytes = parse_audio_frame(pcm_b64, settings.sample_rate)
        except AudioFrameError as e:
            await self.send({"type": "error", "message": str(e)})
            return

        # Accumulate bytes and split into VAD-sized frames
        self._frame_remainder += pcm_bytes
        while len(self._frame_remainder) >= self._vad_frame_bytes:
            frame = self._frame_remainder[: self._vad_frame_bytes]
            self._frame_remainder = self._frame_remainder[self._vad_frame_bytes :]
            await self._process_vad_frame(frame)

    async def _process_vad_frame(self, frame: bytes) -> None:
        """Process a single VAD-sized frame."""
        self._total_frames += 1

        # If VAD is idle, start listening
        if self.vad.state == VADState.IDLE:
            self.vad.start_listening()

        event = self.vad.process_frame(frame)

        # Always add frame to buffer
        self.audio_buffer.add_frame(frame)

        # Handle state transitions
        if event.speech_started:
            # User started speaking
            await self._on_speech_start()

        if self.vad.state == VADState.SPEAKING:
            self._speech_frames += 1
            # Emit partial transcripts periodically
            await self._maybe_emit_partial()

        if event.speech_ended:
            # User stopped speaking
            await self._on_speech_end()

    async def _on_speech_start(self) -> None:
        """Handle speech start: cancel Claude if streaming, start capture."""
        logger.info("Speech started")

        # If Claude is streaming, interrupt it
        if self._is_claude_streaming:
            logger.info("Interrupting Claude stream")
            self._cancel_event.set()
            if self._claude_task and not self._claude_task.done():
                self._claude_task.cancel()
                try:
                    await self._claude_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._is_claude_streaming = False
            self._cancel_event.clear()

        # Reset transcript for new utterance
        self.transcript.reset()
        self._speech_frames = 0
        self._utterance_id += 1  # New utterance scope for partials

        # Start capturing audio
        self.audio_buffer.start_capture()

        # Notify client
        await self.send({"type": "vad_state", "state": "speaking"})

    async def _maybe_emit_partial(self) -> None:
        """
        Kick off a partial transcription if enough audio time has elapsed.

        The actual Whisper call runs in a background task so it never blocks
        the WebSocket receive loop (audio ingestion must stay real-time).
        Only one partial runs at a time.
        """
        now = time.time()
        if now - self._last_partial_time < self._partial_interval_s:
            return

        # Prevent overlapping transcriptions
        if self._partial_running:
            return

        # Snapshot the current speech audio in the event-loop thread (safe read)
        audio = self.audio_buffer.get_speech_audio()
        if audio is None or len(audio) < 8000:  # Less than 0.5s — too short for partial
            return

        self._last_partial_time = now
        self._partial_running = True
        # Fire-and-forget: do not await, so frame ingestion continues.
        asyncio.create_task(self._run_partial(audio, self._utterance_id))

    async def _run_partial(self, audio, utterance_id: int) -> None:
        """Run a single partial transcription in the executor and emit it."""
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                _executor, self.whisper.transcribe_partial, audio
            )
        except Exception as e:
            logger.error(f"Partial transcription error: {e}")
            return
        finally:
            self._partial_running = False

        # Drop the result if the utterance already ended (stale partial).
        if utterance_id != self._utterance_id:
            return

        # Only emit while still on the same utterance.
        if not self._closed and text and self.transcript.update_partial(text):
            await self.send({"type": "partial_transcript", "text": text})

    async def _on_speech_end(self) -> None:
        """Handle speech end: finalize transcript and send to Claude."""
        # Invalidate any in-flight partials for this utterance.
        self._utterance_id += 1
        speech_duration = self.audio_buffer.duration_ms()
        logger.info(
            f"Speech ended, finalizing... "
            f"({speech_duration:.0f}ms, {self._speech_frames} frames)"
        )

        # Stop capture but keep buffer contents
        self.audio_buffer.stop_capture()
        await self.send({"type": "vad_state", "state": "finalizing"})

        # Final transcription
        audio = self.audio_buffer.get_speech_audio()
        if audio is None:
            logger.warning("No speech audio to transcribe (buffer is None)")
            self._reset_pipeline()
            return

        audio_duration_s = len(audio) / 16000.0
        if audio_duration_s < 0.3:
            logger.warning(f"Speech audio too short ({audio_duration_s:.2f}s), discarding")
            self._reset_pipeline()
            await self.send({"type": "vad_state", "state": "listening"})
            return

        logger.info(f"Running final transcription on {audio_duration_s:.2f}s of audio...")

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                _executor, self.whisper.transcribe_buffer, audio
            )
        except Exception as e:
            logger.error(f"Final transcription error: {e}")
            await self.send({"type": "error", "message": f"Transcription failed: {e}"})
            self._reset_pipeline()
            return

        if not text:
            fallback_text = self.transcript.partial_text.strip()
            if fallback_text:
                logger.warning(
                    f"Final transcription empty; using last partial transcript "
                    f"from {audio_duration_s:.2f}s audio"
                )
                text = fallback_text
            else:
                logger.warning(
                    f"Empty transcription from {audio_duration_s:.2f}s audio — "
                    "mic may be too quiet or noisy"
                )
                self._reset_pipeline()
                await self.send({"type": "vad_state", "state": "listening"})
                return

        # Commit final transcript
        self.transcript.update_partial(text)
        final_text = self.transcript.finalize()
        await self.send({"type": "final_transcript", "text": final_text})

        # Send to Claude
        self._cancel_event.clear()
        self._claude_task = asyncio.create_task(
            self._stream_claude_response(final_text)
        )

        # Reset audio pipeline for next utterance (but keep listening)
        self._reset_pipeline()

    async def _stream_claude_response(self, text: str) -> None:
        """Stream Claude's response to the client."""
        self._is_claude_streaming = True

        try:
            async for delta in self.claude.stream_answer(
                user_text=text,
                cancel_event=self._cancel_event,
            ):
                await self.send({"type": "claude_delta", "text": delta})

            if not self._cancel_event.is_set():
                await self.send({"type": "claude_done"})
                logger.info("Claude response complete")
        except asyncio.CancelledError:
            logger.info("Claude streaming cancelled")
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            await self.send({"type": "error", "message": f"Claude error: {e}"})
        finally:
            self._is_claude_streaming = False

    def _reset_pipeline(self) -> None:
        """Reset audio buffer and VAD for the next utterance."""
        self.audio_buffer.clear()
        self.vad.reset()
        self.vad.start_listening()
        self.transcript.reset()
        self._frame_remainder = b""
        self._speech_frames = 0

    async def handle_reset(self) -> None:
        """Handle a full session reset."""
        logger.info("Session reset requested")

        # Cancel Claude if streaming
        if self._is_claude_streaming:
            self._cancel_event.set()
            if self._claude_task and not self._claude_task.done():
                self._claude_task.cancel()
            self._is_claude_streaming = False

        self._reset_pipeline()
        self._cancel_event.clear()
        await self.send({"type": "vad_state", "state": "idle"})

    async def handle_stop(self) -> None:
        """Handle graceful stop."""
        logger.info("Session stop requested")
        await self.handle_reset()


async def voice_session_handler(
    websocket: WebSocket,
    whisper: WhisperEngine,
    claude: LLMClient,
) -> None:
    """
    Main WebSocket handler for a voice session.

    Args:
        websocket: The WebSocket connection.
        whisper: Shared Whisper engine.
        claude: Shared Claude client.
    """
    await websocket.accept()
    session = VoiceSession(websocket, whisper, claude)
    logger.info("Voice session connected")

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            if msg_type == "audio_frame":
                pcm = msg.get("pcm", "")
                if pcm:
                    await session.handle_audio_frame(pcm)

            elif msg_type == "reset":
                await session.handle_reset()

            elif msg_type == "stop":
                await session.handle_stop()
                break

            else:
                await session.send({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("Voice session disconnected")
    except RuntimeError as e:
        # Starlette raises RuntimeError when WebSocket is already closed
        if "WebSocket is not connected" in str(e) or "disconnect" in str(e).lower():
            logger.info(f"Voice session disconnected (runtime): {e}")
        else:
            logger.error(f"Voice session runtime error: {e}")
    except Exception as e:
        logger.error(f"Voice session error: {e}")
    finally:
        session._closed = True
        # Cleanup
        if session._is_claude_streaming:
            session._cancel_event.set()
            if session._claude_task and not session._claude_task.done():
                session._claude_task.cancel()
        logger.info("Voice session closed")
