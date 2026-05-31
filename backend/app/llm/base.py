"""
Common interface for LLM backends.

Both the direct Anthropic Messages API client (`ClaudeClient`) and the
native Claude Code CLI client (`ClaudeCodeClient`) implement this protocol,
so the WebSocket session can use either one transparently.
"""

import asyncio
from typing import AsyncGenerator, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """A streaming text backend driven by a finalized voice transcript."""

    async def stream_answer(
        self,
        user_text: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response to ``user_text`` as text deltas.

        Implementations must stop early if ``cancel_event`` is set
        (used for barge-in / interruption).
        """
        ...
