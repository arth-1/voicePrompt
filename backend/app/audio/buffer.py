"""
Audio ring buffer with pre-roll support.

Stores raw PCM bytes and provides methods to retrieve accumulated
speech audio as a float32 numpy array for Whisper transcription.
"""

from collections import deque
from typing import Optional

import numpy as np

from .frames import pcm_to_float32
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Default pre-roll: 200ms at 20ms frames = 10 frames
_DEFAULT_PREROLL_FRAMES = 10


class AudioBuffer:
    """
    Ring buffer for accumulating speech audio.

    Maintains two buffers:
    - ``_preroll``: fixed-size deque holding the last N frames before speech
      starts, so the beginning of speech is not clipped.
    - ``_speech_frames``: list of PCM frames accumulated during active speech.

    Args:
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Duration of each frame in milliseconds.
        preroll_frames: Number of frames to keep as pre-roll.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        preroll_frames: int = _DEFAULT_PREROLL_FRAMES,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.bytes_per_frame = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit

        # Pre-roll ring buffer (fixed-size deque auto-evicts oldest)
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)

        # Active speech frames
        self._speech_frames: list[bytes] = []

        # Track whether we are in speech mode
        self._is_capturing = False

    @property
    def is_capturing(self) -> bool:
        """Whether the buffer is in speech-capture mode."""
        return self._is_capturing

    def add_frame(self, frame: bytes) -> None:
        """
        Add a PCM frame to the buffer.

        If capturing, the frame goes to the speech buffer.
        Otherwise it goes to the pre-roll ring.

        Args:
            frame: Raw 16-bit PCM bytes.
        """
        if self._is_capturing:
            self._speech_frames.append(frame)
        else:
            self._preroll.append(frame)

    def start_capture(self) -> None:
        """
        Begin speech capture.

        Moves all pre-roll frames into the speech buffer so the
        start of speech is preserved.
        """
        if self._is_capturing:
            return
        self._is_capturing = True
        # Prepend pre-roll frames
        self._speech_frames = list(self._preroll) + self._speech_frames
        self._preroll.clear()
        logger.debug(
            f"Capture started with {len(self._speech_frames)} pre-roll frames"
        )

    def stop_capture(self) -> None:
        """Stop speech capture without clearing the buffer."""
        self._is_capturing = False

    def get_speech_audio(self) -> Optional[np.ndarray]:
        """
        Get all accumulated speech audio as a float32 numpy array.

        Returns:
            Float32 array normalized to [-1.0, 1.0], or ``None`` if empty.
        """
        if not self._speech_frames:
            return None
        pcm = b"".join(self._speech_frames)
        return pcm_to_float32(pcm)

    def get_speech_pcm(self) -> bytes:
        """
        Get all accumulated speech audio as raw PCM bytes.

        Returns:
            Concatenated raw PCM bytes of all speech frames.
        """
        return b"".join(self._speech_frames)

    def duration_ms(self) -> float:
        """
        Total duration of accumulated speech audio in milliseconds.

        Returns:
            Duration in ms.
        """
        total_bytes = sum(len(f) for f in self._speech_frames)
        total_samples = total_bytes / 2  # 16-bit = 2 bytes per sample
        return (total_samples / self.sample_rate) * 1000

    def clear(self) -> None:
        """Clear all buffers and reset state."""
        self._preroll.clear()
        self._speech_frames.clear()
        self._is_capturing = False

    def frame_count(self) -> int:
        """Number of speech frames currently buffered."""
        return len(self._speech_frames)
