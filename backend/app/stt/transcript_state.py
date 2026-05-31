"""
Transcript state tracking with partial and final text management.

Provides a simple diff mechanism to avoid re-sending unchanged
partial transcripts to the client.
"""

from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptState:
    """
    Tracks the current utterance's partial and final transcript.

    Usage:
        state = TranscriptState()
        changed = state.update_partial("hello wor")  # True
        changed = state.update_partial("hello wor")  # False (no change)
        changed = state.update_partial("hello world") # True
        final = state.finalize()  # "hello world"
    """

    def __init__(self):
        self._partial_text: str = ""
        self._final_text: str = ""
        self._is_finalized: bool = False

    @property
    def partial_text(self) -> str:
        """Current partial transcript text."""
        return self._partial_text

    @property
    def final_text(self) -> str:
        """Final committed transcript text."""
        return self._final_text

    @property
    def is_finalized(self) -> bool:
        """Whether the current utterance has been finalized."""
        return self._is_finalized

    def update_partial(self, text: str) -> bool:
        """
        Update the partial transcript.

        Args:
            text: New partial transcription text.

        Returns:
            ``True`` if the text actually changed, ``False`` otherwise.
        """
        text = text.strip()
        if text == self._partial_text:
            return False
        self._partial_text = text
        return True

    def finalize(self) -> str:
        """
        Commit the current partial text as the final transcript.

        Returns:
            The final transcript text.
        """
        self._final_text = self._partial_text
        self._is_finalized = True
        logger.info(f"Transcript finalized: '{self._final_text[:100]}'")
        return self._final_text

    def reset(self) -> None:
        """Clear all state for the next utterance."""
        self._partial_text = ""
        self._final_text = ""
        self._is_finalized = False

    def has_content(self) -> bool:
        """Whether there is any partial text accumulated."""
        return len(self._partial_text) > 0
