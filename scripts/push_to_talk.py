"""
Universal Push-to-Talk Dictation launcher.

Hold the global hotkey (default: ctrl+shift+space) anywhere in Windows, speak,
and release. Your speech is transcribed and typed at the current cursor
position — in Word, Chrome, a terminal, an editor, any text field.

Usage:
    python scripts/push_to_talk.py

Configure via backend/.env:
    PTT_HOTKEY=ctrl+shift+space
    PTT_OUTPUT_MODE=type        # or 'paste'
    WHISPER_MODEL=small

Note: synthesizing keystrokes / global hotkeys may require running this
terminal "as administrator" on some systems (especially to type into other
elevated apps).

Press Ctrl+C to quit.
"""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from app.dictation import main  # noqa: E402

if __name__ == "__main__":
    main()
