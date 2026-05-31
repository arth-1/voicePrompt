"""
Timing utilities for latency measurement.

Usage:
    from app.utils.timing import timed, Timer

    @timed("whisper_transcribe")
    def transcribe(audio): ...

    with Timer("claude_first_token") as t:
        ...
    print(t.elapsed_ms)
"""

import functools
import time
from typing import Any, Callable, Optional

from .logging import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Context manager that measures wall-clock elapsed time.

    Attributes:
        label: Human-readable label for log output.
        elapsed_ms: Elapsed time in milliseconds (set after ``__exit__``).
    """

    def __init__(self, label: str = "operation"):
        self.label = label
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        logger.debug(f"⏱  {self.label}: {self.elapsed_ms:.1f} ms")


def timed(label: Optional[str] = None) -> Callable:
    """
    Decorator that logs the execution time of a function.

    Args:
        label: Optional label; defaults to the function's qualified name.
    """

    def decorator(func: Callable) -> Callable:
        _label = label or func.__qualname__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(_label):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"⏱  {_label}: {elapsed:.1f} ms")

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
