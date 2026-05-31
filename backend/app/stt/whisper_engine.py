"""
Whisper speech-to-text engine using faster-whisper.

Owns the Whisper model and provides transcription of audio buffers.
The model is loaded once at init and reused for all transcriptions.
Thread-safe usage is achieved by running transcription in an executor
from async context.
"""

from typing import Optional, Tuple

import numpy as np

from ..utils.logging import get_logger
from ..utils.timing import Timer

logger = get_logger(__name__)


class WhisperEngine:
    """
    faster-whisper transcription engine.

    Args:
        model_name: Model size name (e.g., 'small', 'medium', 'distil-large-v3').
        device: Compute device ('cpu' or 'cuda').
        compute_type: CTranslate2 compute type ('int8', 'float16', 'float32').
    """

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        logger.info(
            f"Loading Whisper model '{model_name}' on {device} ({compute_type})..."
        )
        with Timer("whisper_model_load"):
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        logger.info(f"Whisper model '{model_name}' loaded successfully")

    def transcribe_buffer(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 audio buffer (final/high-quality mode).

        Uses a slightly wider beam and disables Whisper's internal VAD because
        the audio has already been segmented by the external WebRTC VAD.

        Args:
            audio: Float32 numpy array, mono, 16kHz, normalized to [-1.0, 1.0].

        Returns:
            Transcribed text string.
        """
        if audio is None or len(audio) == 0:
            logger.warning("transcribe_buffer called with empty audio")
            return ""

        duration_s = len(audio) / 16000.0
        logger.info(f"Final transcription: {duration_s:.2f}s of audio ({len(audio)} samples)")

        if duration_s < 0.3:
            logger.warning(f"Audio too short ({duration_s:.2f}s), skipping")
            return ""

        with Timer("whisper_transcribe_final"):
            segments, info = self.model.transcribe(
                audio,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
                language="en",
            )
            # Consume all segments (generator)
            all_segments = list(segments)
            text = " ".join(seg.text.strip() for seg in all_segments if seg.text.strip())

        if text:
            logger.info(
                f"Final transcript ({info.duration:.1f}s, {len(all_segments)} segments): '{text}'"
            )
        else:
            logger.warning(
                f"Empty final transcript from {duration_s:.2f}s audio "
                f"({len(all_segments)} segments, language={info.language}, prob={info.language_probability:.2f})"
            )

        return text

    def transcribe_partial(self, audio: np.ndarray) -> str:
        """
        Transcribe for partial/live updates.

        Uses beam_size=1 and no vad_filter for maximum speed.

        Args:
            audio: Float32 numpy array.

        Returns:
            Partial transcription text.
        """
        if audio is None or len(audio) == 0:
            return ""

        duration_s = len(audio) / 16000.0
        if duration_s < 0.5:
            # Need at least 0.5s for meaningful partial transcription
            return ""

        with Timer("whisper_transcribe_partial"):
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
                language="en",
            )
            all_segments = list(segments)
            text = " ".join(seg.text.strip() for seg in all_segments if seg.text.strip())

        if text:
            logger.debug(f"Partial transcript ({duration_s:.1f}s): '{text[:80]}'")

        return text
