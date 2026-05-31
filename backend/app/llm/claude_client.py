"""
Claude client using the Anthropic Python SDK with async streaming.

Provides an async generator that yields text deltas as Claude
streams its response, supporting cancellation via asyncio.Event.
"""

import asyncio
from typing import AsyncGenerator, Optional

from anthropic import AsyncAnthropic

from .prompts import SYSTEM_PROMPT, format_user_message
from ..utils.logging import get_logger
from ..utils.timing import Timer

logger = get_logger(__name__)


class ClaudeClient:
    """
    Async Claude client for streaming responses.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier (e.g., 'claude-haiku-4-5').
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        if not api_key:
            logger.warning(
                "No ANTHROPIC_API_KEY set — Claude calls will fail. "
                "Set it in your .env file or environment."
            )
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None
        self.model = model
        logger.info(f"Claude client initialized with model '{model}'")

    async def stream_answer(
        self,
        user_text: str,
        cancel_event: Optional[asyncio.Event] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Stream Claude's response to a user transcript.

        Yields text deltas as they arrive. Stops early if
        ``cancel_event`` is set (for interruption support).

        Args:
            user_text: The finalized voice transcript.
            cancel_event: Optional event; if set, streaming stops.
            system_prompt: System prompt for Claude.
            max_tokens: Maximum response tokens.

        Yields:
            Text delta strings.
        """
        formatted_message = format_user_message(user_text)
        logger.info(f"Sending to Claude ({self.model}): '{user_text[:80]}'")

        if self.client is None:
            yield "[Claude unavailable: API key not configured. Transcript saved.]"
            return

        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_message,
                    }
                ],
            ) as stream:
                first_token = True
                timer_start = asyncio.get_event_loop().time()

                async for text in stream.text_stream:
                    # Check for cancellation (user started speaking again)
                    if cancel_event and cancel_event.is_set():
                        logger.info("Claude stream cancelled by user interruption")
                        return

                    if first_token:
                        latency = (asyncio.get_event_loop().time() - timer_start) * 1000
                        logger.info(f"Claude first token latency: {latency:.0f}ms")
                        first_token = False

                    yield text

        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield "[Claude unavailable: response could not be streamed.]"
