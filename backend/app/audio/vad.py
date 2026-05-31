"""
Voice Activity Detection using WebRTC VAD.

Implements a state machine for speech boundary detection:
    IDLE → LISTENING → SPEAKING → COOLDOWN → FINALIZING

Features:
- Configurable aggressiveness (0–3).
- Hangover / cooldown to prevent premature cutoff.
- Minimum speech duration to avoid false triggers.
- 20ms frame processing (640 bytes at 16kHz).

Timing note:
    Speech and silence durations are measured by *counting frames*, not by
    wall-clock time. Each processed frame represents exactly
    ``frame_duration_ms`` of audio. This keeps turn detection deterministic and
    immune to network / CPU scheduling jitter — frames may arrive in bursts
    (e.g. while a transcription is running) but the audio-duration math stays
    correct.
"""

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import webrtcvad

from ..utils.logging import get_logger

logger = get_logger(__name__)


class VADState(str, enum.Enum):
    """VAD state machine states."""

    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    COOLDOWN = "cooldown"
    FINALIZING = "finalizing"


@dataclass
class VADEvent:
    """Event emitted by the VAD on state transitions."""

    previous_state: VADState
    new_state: VADState
    is_speech: bool
    timestamp: float = field(default_factory=time.time)

    @property
    def speech_started(self) -> bool:
        return self.new_state == VADState.SPEAKING and self.previous_state != VADState.SPEAKING

    @property
    def speech_ended(self) -> bool:
        return self.new_state == VADState.FINALIZING


class VADDetector:
    """
    WebRTC VAD wrapper with state machine for speech boundary detection.

    Args:
        mode: Aggressiveness mode (0=least, 3=most).
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Frame duration (10, 20, or 30 ms).
        max_silence_ms: Silence duration before finalizing.
        min_speech_ms: Minimum speech duration to commit.
    """

    def __init__(
        self,
        mode: int = 2,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        max_silence_ms: int = 700,
        min_speech_ms: int = 300,
    ):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.max_silence_ms = max_silence_ms
        self.min_speech_ms = min_speech_ms

        # Expected frame size in bytes (16-bit mono)
        self.frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2

        # Silence / speech thresholds expressed in *frames* (deterministic timing)
        self._max_silence_frames = max(1, math.ceil(max_silence_ms / frame_duration_ms))
        self._min_speech_frames = max(1, math.ceil(min_speech_ms / frame_duration_ms))

        # State
        self._state = VADState.IDLE
        self._speech_frame_count = 0   # frames since speech started
        self._silence_frame_count = 0  # consecutive non-speech frames

    @property
    def state(self) -> VADState:
        """Current VAD state."""
        return self._state

    def reset(self) -> None:
        """Reset the VAD to idle state."""
        self._state = VADState.IDLE
        self._speech_frame_count = 0
        self._silence_frame_count = 0

    def start_listening(self) -> None:
        """Transition to listening state."""
        self._state = VADState.LISTENING
        logger.debug("VAD: LISTENING")

    def process_frame(self, frame: bytes) -> VADEvent:
        """
        Process a single audio frame and return a VAD event.

        The frame must be exactly ``self.frame_bytes`` bytes of
        16-bit mono PCM at the configured sample rate.

        Speech/silence durations are tracked by counting frames, so the
        result is independent of how fast frames are delivered.

        Args:
            frame: Raw PCM audio frame.

        Returns:
            VADEvent describing any state transition.
        """
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"Frame size {len(frame)} != expected {self.frame_bytes} bytes "
                f"({self.frame_duration_ms}ms at {self.sample_rate}Hz)"
            )

        is_speech = self.vad.is_speech(frame, self.sample_rate)
        prev_state = self._state

        if self._state == VADState.IDLE:
            # Do nothing until explicitly started
            pass

        elif self._state == VADState.LISTENING:
            if is_speech:
                self._state = VADState.SPEAKING
                self._speech_frame_count = 1
                self._silence_frame_count = 0
                logger.debug("VAD: SPEAKING (speech detected)")

        elif self._state == VADState.SPEAKING:
            self._speech_frame_count += 1
            if is_speech:
                self._silence_frame_count = 0
            else:
                self._silence_frame_count += 1
                silence_ms = self._silence_frame_count * self.frame_duration_ms

                if self._silence_frame_count >= self._max_silence_frames:
                    speech_ms = self._speech_frame_count * self.frame_duration_ms
                    if self._speech_frame_count >= self._min_speech_frames:
                        self._state = VADState.FINALIZING
                        logger.debug(
                            f"VAD: FINALIZING (silence={silence_ms:.0f}ms, "
                            f"speech={speech_ms:.0f}ms)"
                        )
                    else:
                        # Too short, discard and go back to listening
                        self._state = VADState.LISTENING
                        self._speech_frame_count = 0
                        self._silence_frame_count = 0
                        logger.debug(
                            f"VAD: Discarded short utterance ({speech_ms:.0f}ms)"
                        )

        elif self._state == VADState.COOLDOWN:
            if is_speech:
                # User started speaking again during cooldown
                self._state = VADState.SPEAKING
                self._speech_frame_count = 1
                self._silence_frame_count = 0
                logger.debug("VAD: SPEAKING (resumed from cooldown)")
            # Otherwise stay in cooldown

        elif self._state == VADState.FINALIZING:
            # Stay finalizing until explicitly reset
            pass

        return VADEvent(
            previous_state=prev_state,
            new_state=self._state,
            is_speech=is_speech,
        )

    def speech_duration_ms(self) -> float:
        """Get duration of current speech segment in milliseconds."""
        return self._speech_frame_count * self.frame_duration_ms
