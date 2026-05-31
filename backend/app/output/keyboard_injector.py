"""
Keystroke injection for typing the final transcript into the currently
focused window — e.g. an already-running interactive Claude Code terminal.

This is how voice drives Claude Code *without* starting a new session or
consuming Agent-SDK / `claude -p` credits: we simply act like a keyboard and
type the transcript into the session you already have open, then press Enter.

Strategy (Windows):
- Put the transcript on the clipboard and paste it (Ctrl+V), then send Enter.
  Pasting is far more reliable than synthesizing one WM_KEYDOWN per character
  for a terminal TUI, and it avoids per-character timing issues. It also means
  text starting with '/' doesn't get typed slowly enough to flicker Claude
  Code's slash-command menu.
- Uses only Win32 via ctypes + PowerShell for clipboard, so there are no extra
  Python dependencies.

The injector is intentionally simple and best-effort: if no suitable target
window is focused, it still types into whatever is focused. Use
``target_title_contains`` to refuse injection unless the focused window title
matches (a safety guard so you don't dump a transcript into the wrong app).
"""

from __future__ import annotations

import ctypes
import subprocess
import time
from ctypes import wintypes
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ── Win32 setup ────────────────────────────────────────────────────────
_user32 = ctypes.WinDLL("user32", use_last_error=True)

# SendInput structures
ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

VK_CONTROL = 0x11
VK_RETURN = 0x0D
VK_V = 0x56


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


def _send_key(vk: int, up: bool = False) -> None:
    flags = KEYEVENTF_KEYUP if up else 0
    inp = _INPUT(
        type=INPUT_KEYBOARD,
        u=_INPUTunion(ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)),
    )
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _tap(vk: int) -> None:
    _send_key(vk, up=False)
    _send_key(vk, up=True)


def _set_clipboard_text(text: str) -> bool:
    """Set clipboard text via PowerShell (handles Unicode safely)."""
    try:
        # Pass text through stdin to avoid any command-line quoting issues.
        proc = subprocess.run(
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
        if proc.returncode != 0:
            logger.error(f"Set-Clipboard failed: {proc.stderr.strip()}")
            return False
        return True
    except Exception as e:
        logger.error(f"Clipboard set error: {e}")
        return False


def _get_foreground_title() -> str:
    """Return the title of the currently focused window."""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


# EnumWindows callback type
_WNDENUMPROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)


def _find_window_by_title(substring: str) -> Optional[int]:
    """Find the first visible top-level window whose title contains substring."""
    substring_l = substring.lower()
    match: list[int] = []

    def _cb(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        if substring_l in buf.value.lower():
            match.append(hwnd)
            return False  # stop enumeration
        return True

    _user32.EnumWindows(_WNDENUMPROC(_cb), 0)
    return match[0] if match else None


def _focus_window(hwnd: int) -> bool:
    """
    Bring a window to the foreground.

    Windows blocks SetForegroundWindow from a process that doesn't own the
    current foreground window (foreground lock). The standard workaround is to
    temporarily attach our thread's input to the foreground window's thread.
    """
    try:
        SW_RESTORE = 9
        _user32.ShowWindow(hwnd, SW_RESTORE)

        fg = _user32.GetForegroundWindow()
        if fg == hwnd:
            return True

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        cur_thread = kernel32.GetCurrentThreadId()
        fg_thread = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
        tgt_thread = _user32.GetWindowThreadProcessId(hwnd, None)

        attached_fg = attached_tgt = False
        if fg_thread and fg_thread != cur_thread:
            attached_fg = bool(_user32.AttachThreadInput(cur_thread, fg_thread, True))
        if tgt_thread and tgt_thread != cur_thread and tgt_thread != fg_thread:
            attached_tgt = bool(_user32.AttachThreadInput(cur_thread, tgt_thread, True))

        try:
            _user32.BringWindowToTop(hwnd)
            _user32.SetForegroundWindow(hwnd)
            _user32.SetActiveWindow(hwnd)
        finally:
            if attached_fg:
                _user32.AttachThreadInput(cur_thread, fg_thread, False)
            if attached_tgt:
                _user32.AttachThreadInput(cur_thread, tgt_thread, False)

        return _user32.GetForegroundWindow() == hwnd
    except Exception as e:
        logger.error(f"Focus window error: {e}")
        return False


class KeyboardInjector:
    """
    Types text into the currently focused window via clipboard paste + Enter.

    Args:
        press_enter: Whether to send Enter after pasting (submits the prompt).
        target_title_contains: If set, injection targets the window whose title
            contains this substring (case-insensitive). Use it to ensure you
            only ever type into your Claude Code terminal.
        paste_delay_s: Small delay between paste and Enter so the TUI registers
            the pasted text before submission.
        auto_focus: If True and the target window isn't focused, find it by
            title and bring it to the foreground before pasting.
    """

    def __init__(
        self,
        press_enter: bool = True,
        target_title_contains: str = "",
        paste_delay_s: float = 0.15,
        auto_focus: bool = True,
    ):
        self.press_enter = press_enter
        self.target_title_contains = target_title_contains.strip()
        self.paste_delay_s = paste_delay_s
        self.auto_focus = auto_focus

    def is_target_focused(self) -> bool:
        """Whether the current foreground window matches the configured guard."""
        if not self.target_title_contains:
            return True
        title = _get_foreground_title()
        return self.target_title_contains.lower() in title.lower()

    def _ensure_target(self) -> bool:
        """
        Make sure the right window is focused.

        - No guard configured: always OK (types into whatever is focused).
        - Guard configured and already focused: OK.
        - Guard configured, not focused, auto_focus on: try to find & focus it.
        """
        if not self.target_title_contains:
            return True
        if self.is_target_focused():
            return True
        if not self.auto_focus:
            return False

        hwnd = _find_window_by_title(self.target_title_contains)
        if hwnd is None:
            logger.warning(
                f"No window titled like '{self.target_title_contains}' found to focus"
            )
            return False
        if _focus_window(hwnd):
            time.sleep(0.12)  # let focus settle
            return self.is_target_focused()
        return False

    def inject(self, text: str) -> bool:
        """
        Inject ``text`` into the target (Claude Code) window.

        Returns True if the keystrokes were sent, False if skipped/failed.
        """
        text = text.strip()
        if not text:
            return False

        if not self._ensure_target():
            logger.warning(
                f"Skipping injection — could not focus target "
                f"'{self.target_title_contains}' (focused: '{_get_foreground_title()}')"
            )
            return False

        if not _set_clipboard_text(text):
            return False

        # Ctrl+V
        _send_key(VK_CONTROL, up=False)
        _tap(VK_V)
        _send_key(VK_CONTROL, up=True)

        if self.press_enter:
            time.sleep(self.paste_delay_s)
            _tap(VK_RETURN)

        logger.info(f"Injected transcript into focused window: '{text[:80]}'")
        return True
