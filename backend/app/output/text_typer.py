"""
Type text at the current cursor position in *any* Windows application.

Used by the universal push-to-talk dictation mode: whatever app has focus
(Word, Chrome, a terminal, an editor) receives the transcribed speech as if it
were typed on the keyboard.

Two strategies:
- ``type``  (default): synthesize Unicode keystrokes via SendInput. Works in
  virtually every text field and never touches the clipboard.
- ``paste``: copy text to the clipboard and send Ctrl+V (faster for long text).
  The previous clipboard contents are saved and restored afterwards.

Windows-only (Win32 SendInput). On other platforms ``type_text`` is a no-op
that returns False.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from ctypes import wintypes
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

_IS_WIN = sys.platform == "win32"

if _IS_WIN:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)

    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    INPUT_KEYBOARD = 1

    VK_CONTROL = 0x11
    VK_V = 0x56

    # Modifier virtual-key codes we force-release before typing, so a
    # physically-held hotkey modifier (ctrl/shift/alt/win) can't turn our
    # injected characters into control-combos that the target app discards.
    _MODIFIER_VKS = (
        0x10, 0x11, 0x12,        # generic SHIFT, CONTROL, ALT(MENU)
        0xA0, 0xA1,              # L/R SHIFT
        0xA2, 0xA3,              # L/R CONTROL
        0xA4, 0xA5,              # L/R ALT(MENU)
        0x5B, 0x5C,              # L/R WIN
    )

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class _INPUTunion(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]

    def _send_inputs(inputs: list) -> None:
        n = len(inputs)
        arr = (_INPUT * n)(*inputs)
        _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))

    def _vk_input(vk: int, up: bool) -> _INPUT:
        flags = KEYEVENTF_KEYUP if up else 0
        return _INPUT(
            type=INPUT_KEYBOARD,
            u=_INPUTunion(
                ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)
            ),
        )

    def _unicode_inputs(ch: str) -> list:
        """Build keydown+keyup INPUTs for a single UTF-16 code unit."""
        code = ord(ch)
        down = _INPUT(
            type=INPUT_KEYBOARD,
            u=_INPUTunion(
                ki=_KEYBDINPUT(
                    wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=None
                )
            ),
        )
        up = _INPUT(
            type=INPUT_KEYBOARD,
            u=_INPUTunion(
                ki=_KEYBDINPUT(
                    wVk=0,
                    wScan=code,
                    dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                    time=0,
                    dwExtraInfo=None,
                )
            ),
        )
        return [down, up]

    def _release_held_modifiers() -> list:
        """
        Send key-up events for all modifier keys before typing.

        The push-to-talk hotkey (e.g. ctrl+shift+space) means ctrl/shift may be
        physically down when we start typing. If they stay down, the target app
        interprets our injected characters as Ctrl+<char> / Shift+<char> and
        drops or mangles them.

        We send key-ups unconditionally: releasing a modifier that isn't down
        is a harmless no-op, so this is more reliable than trying to detect the
        held state (injected/physical state isn't always visible to us).
        """
        ups = [_vk_input(vk, True) for vk in _MODIFIER_VKS]
        _send_inputs(ups)
        return list(_MODIFIER_VKS)
else:
    def _release_held_modifiers() -> list:  # type: ignore
        return []


def _get_clipboard() -> Optional[str]:
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return p.stdout if p.returncode == 0 else None
    except Exception:
        return None


def _set_clipboard(text: str) -> bool:
    try:
        p = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$t = [Console]::In.ReadToEnd(); Set-Clipboard -Value $t",
            ],
            input=text,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return p.returncode == 0
    except Exception as e:
        logger.error(f"Clipboard set failed: {e}")
        return False


class TextTyper:
    """
    Types text into the focused window.

    Args:
        mode: 'type' (Unicode keystrokes, clipboard-safe) or 'paste'
            (clipboard + Ctrl+V, faster for long text).
        char_delay_s: Optional per-chunk delay in 'type' mode. 0 = fastest.
        restore_clipboard: In 'paste' mode, restore the previous clipboard
            contents after pasting.
    """

    def __init__(
        self,
        mode: str = "type",
        char_delay_s: float = 0.0,
        restore_clipboard: bool = True,
    ):
        self.mode = mode.strip().lower()
        self.char_delay_s = char_delay_s
        self.restore_clipboard = restore_clipboard

    def type_text(self, text: str) -> bool:
        """Emit ``text`` at the current cursor. Returns True on success."""
        if not text:
            return False
        if not _IS_WIN:
            logger.warning("Text typing is only implemented on Windows.")
            return False

        # Release any physically-held hotkey modifiers (ctrl/shift/alt/win) so
        # they don't turn our injected text into control-combos and get it
        # dropped by the target app.
        _release_held_modifiers()

        if self.mode == "paste":
            return self._paste(text)
        return self._type_unicode(text)

    def _type_unicode(self, text: str) -> bool:
        """Send the text as Unicode keystrokes (handles surrogate pairs)."""
        try:
            chunk: list = []
            for ch in text:
                # Split this character into UTF-16 code units so non-BMP
                # characters (e.g. emoji) are sent as surrogate pairs.
                utf16 = ch.encode("utf-16-le")
                for i in range(0, len(utf16), 2):
                    code_unit = utf16[i] | (utf16[i + 1] << 8)
                    chunk += _unicode_inputs(chr(code_unit))

                if self.char_delay_s:
                    _send_inputs(chunk)
                    chunk = []
                    time.sleep(self.char_delay_s)

            if chunk:
                # Send in reasonably sized batches to avoid huge single calls.
                BATCH = 200
                for i in range(0, len(chunk), BATCH):
                    _send_inputs(chunk[i : i + BATCH])
            logger.info(f"Typed {len(text)} chars at cursor.")
            return True
        except Exception as e:
            logger.error(f"Unicode typing failed: {e}")
            return False

    def _paste(self, text: str) -> bool:
        prev = _get_clipboard() if self.restore_clipboard else None
        if not _set_clipboard(text):
            return False
        try:
            _send_inputs([_vk_input(VK_CONTROL, False), _vk_input(VK_V, False),
                          _vk_input(VK_V, True), _vk_input(VK_CONTROL, True)])
            logger.info(f"Pasted {len(text)} chars at cursor.")
        except Exception as e:
            logger.error(f"Paste failed: {e}")
            return False
        finally:
            if prev is not None:
                time.sleep(0.25)  # let the paste complete before restoring
                _set_clipboard(prev)
        return True
