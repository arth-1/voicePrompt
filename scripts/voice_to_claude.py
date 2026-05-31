"""
Voice → Claude Code launcher.

Speak into your mic; the transcript is typed into your already-open
interactive Claude Code terminal (same session every time). No browser,
no new Claude session, no `claude -p` credits.

Usage:
    # type into whatever window is focused (focus your Claude Code terminal):
    python scripts/voice_to_claude.py

    # or target a specific terminal window by title via backend/.env:
    #   CC_WINDOW_TITLE=claude
    #   CC_AUTO_FOCUS=true

Press Ctrl+C to stop.
"""

import sys
from pathlib import Path

# Make the backend package importable
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from app.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
