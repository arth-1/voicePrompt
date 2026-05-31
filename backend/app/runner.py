"""
Browser-free voice runner: microphone → VAD → Whisper → Claude Code terminal.

This is the "voice coding through the terminal" path. It captures your mic
locally (no browser, no WebSocket), transcribes speech with faster-whisper, and
types the finalized transcript straight into your already-open interactive
Claude Code session via keystroke injection — same session for every prompt,
no new process, no `claude -p` credits consumed.

Run:
    python scripts/voice_to_claude.py

Speak, pause, and the transcript is pasted into the focused Claude Code
terminal followed by Enter. Press Ctrl+C to stop.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

from .audio.buffer import AudioBuffer
from .audio.vad import VADDetector, VADState
from .config import settings
from .output.keyboard_injector import KeyboardInjector
from .stt.transcript_state import TranscriptState
from .stt.whisper_engine import WhisperEngine
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_FRAME_MS = 20


class VoiceRunner:
    """
    Local mic-to-Claude-Code loop.

    Args:
        whisper: Loaded Whisper engine.
        injector: Keyboard injector targeting the Claude Code terminal.
        partial_console: If True, print live partial transcripts to the console.
    """

    def __init__(
        self,
        whisper: WhisperEngine,
        injector: KeyboardInjector,
        partial_console: bool = True,
    ):
        self.whisper = whisper
        self.injector = injector
        self.partial_console = partial_console

        self.sample_rate = settings.sample_rate
        self.frame_bytes = int(self.sample_rate * _FRAME_MS / 1000) * 2  # 640

        self.vad = VADDetector(
            mode=settings.vad_mode,
            sample_rate=self.sample_rate,
            frame_duration_ms=_FRAME_MS,
            max_silence_ms=settings.max_silence_ms,
            min_speech_ms=settings.min_speech_ms,
        )
        self.buffer = AudioBuffer(
            sample_rate=self.sample_rate,
            frame_duration_ms=_FRAME_MS,
        )
        self.transcript = TranscriptState()

        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._running = False
        self._frame_remainder = b""

        # Partial throttling
        self._last_partial = 0.0
        self._partial_interval = settings.partial_update_ms / 1000.0

    # ── Mic capture ────────────────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — runs on a separate thread."""
        if status:
            logger.debug(f"Audio status: {status}")
        # indata is int16 mono; push raw bytes to the queue
        self._audio_q.put(bytes(indata))

    def run(self) -> None:
        """Start capturing and processing until interrupted."""
        import sounddevice as sd

        self._running = True
        self.vad.start_listening()

        target = self.injector.target_title_contains or "(focused window)"
        print("=" * 64)
        print("  Voice → Claude Code   (Ctrl+C to stop)")
        print("=" * 64)
        print(f"  Target window : {target}")
        print(f"  Whisper model : {settings.whisper_model} ({settings.whisper_device})")
        print(f"  Submit (Enter): {self.injector.press_enter}")
        print("-" * 64)
        print("  Speak now. Pause to send the transcript to Claude Code.\n")

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * _FRAME_MS / 1000),
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                self._process_loop()
        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            self._running = False

    def _process_loop(self) -> None:
        """Drain the audio queue and run frames through the pipeline."""
        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self._frame_remainder += chunk
            while len(self._frame_remainder) >= self.frame_bytes:
                frame = self._frame_remainder[: self.frame_bytes]
                self._frame_remainder = self._frame_remainder[self.frame_bytes :]
                self._process_frame(frame)

    def _process_frame(self, frame: bytes) -> None:
        if self.vad.state == VADState.IDLE:
            self.vad.start_listening()

        event = self.vad.process_frame(frame)
        self.buffer.add_frame(frame)

        if event.speech_started:
            self._on_speech_start()

        if self.vad.state == VADState.SPEAKING:
            self._maybe_partial()

        if event.speech_ended:
            self._on_speech_end()

    def _on_speech_start(self) -> None:
        logger.info("Speech started")
        self.transcript.reset()
        self.buffer.start_capture()
        if self.partial_console:
            print("  🎤 listening…", end="\r", flush=True)

    def _maybe_partial(self) -> None:
        now = time.time()
        if now - self._last_partial < self._partial_interval:
            return
        self._last_partial = now

        audio = self.buffer.get_speech_audio()
        if audio is None or len(audio) < 8000:
            return
        text = self.whisper.transcribe_partial(audio)
        if text and self.transcript.update_partial(text) and self.partial_console:
            # Live partial on one console line
            line = f"  📝 {text}"
            print(line[:100].ljust(100), end="\r", flush=True)

    def _on_speech_end(self) -> None:
        self.buffer.stop_capture()
        audio = self.buffer.get_speech_audio()
        self._reset_pipeline_audio()

        if audio is None:
            return
        duration_s = len(audio) / float(self.sample_rate)
        if duration_s < 0.3:
            logger.info(f"Discarded short utterance ({duration_s:.2f}s)")
            return

        text = self.whisper.transcribe_buffer(audio)
        if not text:
            text = self.transcript.partial_text.strip()
        if not text:
            print("  (no speech recognized)".ljust(100))
            return

        print(" " * 100, end="\r")  # clear partial line
        print(f"  ➤ {text}")

        ok = self.injector.inject(text)
        if not ok:
            print("  ⚠ Could not inject into Claude Code (see log). Transcript above.")

    def _reset_pipeline_audio(self) -> None:
        self.buffer.clear()
        self.vad.reset()
        self.vad.start_listening()
        self.transcript.reset()
        self._frame_remainder = self._frame_remainder  # keep partial bytes


def main() -> None:
    setup_logging()

    if sys.platform != "win32":
        logger.warning(
            "Keystroke injection is implemented for Windows. On other platforms "
            "the transcript will print but not be injected."
        )

    logger.info(f"Loading Whisper model '{settings.whisper_model}'…")
    whisper = WhisperEngine(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    injector = KeyboardInjector(
        press_enter=settings.cc_inject_press_enter,
        target_title_contains=settings.cc_window_title,
        auto_focus=settings.cc_auto_focus,
    )

    runner = VoiceRunner(whisper=whisper, injector=injector)
    runner.run()


if __name__ == "__main__":
    main()
