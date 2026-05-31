"""
Audio frame parsing and normalization.

Handles:
- Base64 decoding of PCM frames received over WebSocket.
- Validation of 16-bit mono PCM format.
- Conversion from int16 PCM to float32 for Whisper.
"""

import base64
import struct

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AudioFrameError(Exception):
    """Raised when an audio frame is malformed."""


def decode_base64_pcm(b64_data: str) -> bytes:
    """
    Decode a base64-encoded PCM audio frame.

    Args:
        b64_data: Base64-encoded string of raw 16-bit PCM bytes.

    Returns:
        Raw PCM bytes.

    Raises:
        AudioFrameError: If the data cannot be decoded.
    """
    try:
        return base64.b64decode(b64_data)
    except Exception as e:
        raise AudioFrameError(f"Invalid base64 audio data: {e}") from e


def validate_pcm_frame(pcm_bytes: bytes, expected_sample_rate: int = 16000) -> None:
    """
    Validate that a PCM frame is 16-bit mono.

    A valid frame must have an even number of bytes (each sample = 2 bytes).

    Args:
        pcm_bytes: Raw PCM bytes.
        expected_sample_rate: Expected sample rate (used for logging only).

    Raises:
        AudioFrameError: If the frame has an odd byte count.
    """
    if len(pcm_bytes) == 0:
        raise AudioFrameError("Empty audio frame")
    if len(pcm_bytes) % 2 != 0:
        raise AudioFrameError(
            f"PCM frame has odd byte count ({len(pcm_bytes)}); expected 16-bit (2 bytes/sample)"
        )


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert 16-bit signed integer PCM to float32 in [-1.0, 1.0].

    This is the format expected by faster-whisper.

    Args:
        pcm_bytes: Raw 16-bit mono PCM bytes.

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0].
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def float32_to_pcm(audio: np.ndarray) -> bytes:
    """
    Convert float32 audio back to 16-bit PCM bytes.

    Useful for feeding audio to WebRTC VAD which expects raw PCM.

    Args:
        audio: Float32 numpy array in [-1.0, 1.0].

    Returns:
        Raw 16-bit PCM bytes.
    """
    clamped = np.clip(audio, -1.0, 1.0)
    int_samples = (clamped * 32767).astype(np.int16)
    return int_samples.tobytes()


def parse_audio_frame(b64_data: str, sample_rate: int = 16000) -> bytes:
    """
    Full pipeline: decode base64 → validate → return raw PCM bytes.

    Args:
        b64_data: Base64-encoded PCM audio.
        sample_rate: Expected sample rate.

    Returns:
        Validated raw PCM bytes.
    """
    pcm = decode_base64_pcm(b64_data)
    validate_pcm_frame(pcm, sample_rate)
    return pcm
