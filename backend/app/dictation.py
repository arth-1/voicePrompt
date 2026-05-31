"""
Universal push-to-talk dictation.

Hold a global hotkey anywhere in Windows, speak, release — your speech is
transcribed and typed at the current cursor position. Works in any app that
accepts keyboard input: Word, Chrome, terminals, editors, chat boxes, etc.

Unlike the WebSocket/VAD pipeline, speech boundaries here are defined by the
key press/release, so there's no silence-detection guesswork:

    [hotkey down]  → start recording
    (you speak)
    [hotkey up]    → stop, transcribe the whole clip, type it at the cursor

Run:
    python scripts/push_to_talk.py

Default hotkey: ctrl+shift+space  (configurable via PTT_HOTKEY in backend/.env)
Note: the laptop 'Fn' key cannot be used — it's handled in keyboard firmware
and never reaches the OS, so no software can detect it as a hotkey.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np

from .config import settings
from .output.text_typer import TextTyper
from .stt.whisper_engine import WhisperEngine
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class PushToTalkDictation:
    """
    Global push-to-talk dictation controller.

    Args:
        whisper: Loaded Whisper engine.
        typer: TextTyper that emits transcribed text at the cursor.
        hotkey: Hotkey combo string understood by the `keyboard` library
            (e.g. 'ctrl+alt+space').
        min_record_s: Ignore clips shorter than this (accidental taps).
        append_space: Append a trailing space so consecutive dictations don't
            run together.
    """

    def __init__(
        self,
        whisper: WhisperEngine,
        typer: TextTyper,
        hotkey: str = "ctrl+shift+space",
        min_record_s: float = 0.3,
        append_space: bool = True,
        type_delay_s: float = 0.006,
    ):
        self.whisper = whisper
        self.typer = typer
        self.hotkey = hotkey
        self.min_record_s = min_record_s
        self.append_space = append_space
        self.type_delay_s = type_delay_s
        # 'paste' (atomic, reliable for sentences) or 'type' (per-char). Mirror
        # the TextTyper's configured mode so PTT_OUTPUT_MODE controls both.
        self.output_mode = getattr(typer, "mode", "paste")

        self.sample_rate = settings.sample_rate

        self._recording = False
        self._frames: list[bytes] = []
        self._lock = threading.Lock()
        self._stream = None
        self._keyboard = None  # set in run(); the keyboard module instance
        self._hook_handle = None  # handle for the suppressing trigger hook
        self._trigger_filter = None
        self._trigger_name = ""
        self._modifier_names: list[str] = []

    # ── Mic stream control ─────────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug(f"Audio status: {status}")
        if self._recording:
            with self._lock:
                self._frames.append(bytes(indata))

    def _start_recording(self) -> None:
        if self._recording:
            return
        with self._lock:
            self._frames = []
        self._recording = True
        logger.info("● recording…")
        print("  ● recording… (release to transcribe)", end="\r", flush=True)

    def _emit_text(self, text: str) -> bool:
        """
        Insert ``text`` at the cursor.

        We run a global *suppressing* keyboard hook for the trigger key, so
        before injecting anything we remove that hook (and reinstall it after),
        otherwise our injected keys race against the listener thread and get
        clipped.

        Two delivery modes:
        - 'paste' (default): put the text on the clipboard and send Ctrl+V.
          The whole transcript arrives atomically, so nothing can be dropped —
          this is the reliable choice for full sentences. The previous
          clipboard contents are restored afterwards.
        - 'type': synthesize per-character keystrokes via ``keyboard.write``,
          paced by ``type_delay_s``. Without pacing, SendInput overruns the
          target app's input queue and characters are dropped (e.g.
          "Hello, what is happening?" -> "Hello, ning?").
        """
        kb = self._keyboard
        if kb is None:
            return self.typer.type_text(text)

        self._remove_hook()
        try:
            if self.output_mode == "type":
                kb.write(text, delay=self.type_delay_s, restore_state_after=False)
                logger.info(f"Typed {len(text)} chars at cursor (keyboard.write).")
            else:
                self._paste_via_clipboard(kb, text)
                logger.info(f"Pasted {len(text)} chars at cursor.")
            ok = True
        except Exception as e:
            logger.error(f"Primary emit failed ({e}); falling back to TextTyper")
            ok = self.typer.type_text(text)
        finally:
            # Let injected keystrokes drain from the OS queue before the
            # suppressing hook comes back, so it can't catch their tail.
            time.sleep(0.05)
            self._install_hook()
        return ok

    def _paste_via_clipboard(self, kb, text: str) -> None:
        """Set clipboard to text, send Ctrl+V, then restore prior clipboard."""
        from .output.text_typer import _get_clipboard, _set_clipboard

        prev = _get_clipboard()
        if not _set_clipboard(text):
            raise RuntimeError("could not set clipboard")
        time.sleep(0.03)
        # Release any held hotkey modifiers, then paste. send() sets the
        # library's is_replaying flag so these events bypass any of our hooks.
        kb.send("ctrl+v")
        if prev is not None:
            time.sleep(0.15)  # let the paste complete before restoring
            _set_clipboard(prev)

    def _stop_and_transcribe(self) -> None:
        if not self._recording:
            return
        self._recording = False

        with self._lock:
            pcm = b"".join(self._frames)
            self._frames = []

        print(" " * 60, end="\r")  # clear status line

        # Duration is derived from the actual captured audio (2 bytes/sample,
        # mono), so it's correct regardless of scheduling/wall-clock timing.
        duration = (len(pcm) / 2) / float(self.sample_rate) if pcm else 0.0

        if duration < self.min_record_s or not pcm:
            logger.info(f"Ignored short clip ({duration:.2f}s)")
            return

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe in a worker thread so the hotkey handler returns fast.
        threading.Thread(target=self._transcribe_and_type, args=(audio,), daemon=True).start()

    def _transcribe_and_type(self, audio: np.ndarray) -> None:
        try:
            text = self.whisper.transcribe_buffer(audio)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return
        if not text:
            print("  (no speech recognized)")
            return

        out = text + (" " if self.append_space else "")
        print(f"  ➤ {text}")
        # Give the user a moment to fully release the hotkey before typing.
        time.sleep(0.12)
        self._emit_text(out)

    # ── Run loop ───────────────────────────────────────────────────
    def _decide_trigger(self, event_type: str, mods_down: bool) -> bool:
        """
        Core push-to-talk decision for a trigger-key event.

        Returns True to let the key event pass through to the focused app,
        False to suppress it. Also drives start/stop of recording.

        - down + already recording      -> suppress (kills auto-repeat)
        - down + modifiers held          -> start recording, suppress
        - down + no modifiers            -> allow (plain key, e.g. a space)
        - up + recording                 -> stop+transcribe, suppress
        - up + not recording             -> allow
        """
        if event_type == "down":
            if self._recording:
                return False
            if mods_down:
                self._start_recording()
                return False
            return True

        # key up
        if self._recording:
            self._stop_and_transcribe()
            return False
        return True

    def run(self) -> None:
        import keyboard
        import sounddevice as sd

        self._keyboard = keyboard

        print("=" * 64)
        print("  Universal Push-to-Talk Dictation   (Ctrl+C to quit)")
        print("=" * 64)
        print(f"  Hotkey        : hold  [{self.hotkey}]")
        print(f"  Whisper model : {settings.whisper_model} ({settings.whisper_device})")
        print(f"  Output mode   : {self.typer.mode}")
        print("-" * 64)
        print("  Focus any text field, hold the hotkey, speak, release.\n")

        # Open the mic stream once; we just gate recording with a flag.
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * 0.02),  # 20ms blocks
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

        # ── Hotkey handling ────────────────────────────────────────
        # We suppress the trigger key so it never leaks into the focused app
        # (otherwise Space + its auto-repeat would spam the target while held).
        #
        # A single suppressed hook on the trigger key drives everything:
        #   - trigger DOWN while modifiers held  -> start recording (suppressed)
        #   - trigger DOWN auto-repeat while recording -> suppressed
        #   - trigger UP while recording -> stop + transcribe (suppressed)
        #   - trigger activity with no modifiers -> passes through normally
        modifier_names = [k.strip() for k in self.hotkey.split("+")[:-1]]
        trigger_name = self.hotkey.split("+")[-1].strip()
        self._modifier_names = modifier_names
        self._trigger_name = trigger_name

        def _trigger_filter(ev) -> bool:
            """Return True to allow the key event through, False to suppress."""
            mods_down = (
                all(keyboard.is_pressed(m) for m in modifier_names)
                if modifier_names else True
            )
            return self._decide_trigger(ev.event_type, mods_down)

        self._trigger_filter = _trigger_filter
        self._install_hook()

        try:
            keyboard.wait()  # block forever (until Ctrl+C)
        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()

    # ── Suppressing hook install/remove ────────────────────────────
    def _install_hook(self) -> None:
        """Install the suppressing trigger-key hook."""
        if self._keyboard is None or self._hook_handle is not None:
            return
        self._hook_handle = self._keyboard.hook_key(
            self._trigger_name, self._trigger_filter, suppress=True
        )

    def _remove_hook(self) -> None:
        """Remove the suppressing trigger-key hook (so we can type freely)."""
        if self._hook_handle is None:
            return
        try:
            # hook_key returns a remover closure; calling it removes the hook.
            self._hook_handle()
        except Exception:
            pass
        self._hook_handle = None


def main() -> None:
    setup_logging()

    if sys.platform != "win32":
        logger.warning(
            "Push-to-talk typing is implemented for Windows. The hotkey and "
            "transcription will run, but text won't be injected on this platform."
        )

    logger.info(f"Loading Whisper model '{settings.whisper_model}'…")
    whisper = WhisperEngine(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    typer = TextTyper(
        mode=settings.ptt_output_mode,
        restore_clipboard=True,
    )

    dictation = PushToTalkDictation(
        whisper=whisper,
        typer=typer,
        hotkey=settings.ptt_hotkey,
        min_record_s=settings.ptt_min_record_s,
        append_space=settings.ptt_append_space,
        type_delay_s=settings.ptt_type_delay_s,
    )
    dictation.run()


if __name__ == "__main__":
    main()
