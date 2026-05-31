"""
Structured logging setup for the Voice-Claude Bridge.

Provides a pre-configured logger with colored output and timestamps.
Usage:
    from app.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Server started")
"""

import logging
import sys
from typing import Optional


# ANSI color codes for terminal output
class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


_LEVEL_COLORS = {
    "DEBUG": _Colors.DIM + _Colors.CYAN,
    "INFO": _Colors.GREEN,
    "WARNING": _Colors.YELLOW,
    "ERROR": _Colors.RED,
    "CRITICAL": _Colors.BOLD + _Colors.RED,
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log level names."""

    FORMAT = (
        f"{_Colors.DIM}%(asctime)s{_Colors.RESET} "
        "%(levelcolor)s%(levelname)-8s%(reset)s "
        f"{_Colors.CYAN}%(name)s{_Colors.RESET} "
        "%(message)s"
    )

    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.levelcolor = _LEVEL_COLORS.get(record.levelname, "")
        record.reset = _Colors.RESET
        self._fmt = self.FORMAT
        return super().format(record)


_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with colored console output."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger. Automatically sets up logging on first call.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    setup_logging()
    return logging.getLogger(name or "voice-bridge")
