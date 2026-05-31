"""
Application configuration loaded from environment variables.

All settings have sensible defaults for a CPU-first MVP.
Override via .env file or system environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env from the backend directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class Settings(BaseSettings):
    """Central configuration for the Voice-Claude Bridge."""

    # ── LLM backend selection ───────────────────────────────────────
    llm_backend: str = Field(
        default="api",
        description=(
            "Which LLM backend to use: 'api' for the direct Anthropic Messages "
            "API, or 'claude_code' to drive the native Claude Code CLI (agentic, "
            "with file/command access in claude_code_workdir)."
        ),
    )

    # ── Anthropic / Claude ──────────────────────────────────────────
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Required for Claude calls.",
    )
    claude_model: str = Field(
        default="claude-haiku-4-5",
        description="Claude model identifier.",
    )

    # ── Claude Code CLI backend ─────────────────────────────────────
    claude_code_workdir: str = Field(
        default="",
        description=(
            "Working directory Claude Code operates in. Empty = current "
            "process working directory."
        ),
    )
    claude_code_permission_mode: str = Field(
        default="default",
        description=(
            "Claude Code permission mode: 'default' (asks/limited), "
            "'acceptEdits' (auto-accept file edits), or 'bypassPermissions' "
            "(full autonomy — use with care)."
        ),
    )
    claude_code_allowed_tools: str = Field(
        default="",
        description=(
            "Comma-separated Claude Code tools to allow (e.g. 'Read,Grep,Edit'). "
            "Empty = CLI default."
        ),
    )
    claude_code_max_turns: int = Field(
        default=12,
        description="Maximum agent turns per voice request (safety cap).",
    )

    # ── Whisper STT ─────────────────────────────────────────────────
    whisper_model: str = Field(
        default="small",
        description="faster-whisper model name (tiny/base/small/medium/large-v3/distil-large-v3).",
    )
    whisper_device: str = Field(
        default="cpu",
        description="Inference device: cpu or cuda.",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="CTranslate2 compute type (int8, float16, float32).",
    )

    # ── Audio ───────────────────────────────────────────────────────
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz. Must be 16000 for Whisper.",
    )

    # ── VAD ──────────────────────────────────────────────────────────
    vad_mode: int = Field(
        default=2,
        ge=0,
        le=3,
        description="WebRTC VAD aggressiveness (0=least, 3=most).",
    )
    max_silence_ms: int = Field(
        default=700,
        description="Milliseconds of silence before finalizing an utterance.",
    )
    min_speech_ms: int = Field(
        default=300,
        description="Minimum speech duration to avoid false triggers.",
    )

    # ── Transcription ────────────────────────────────────────────────
    partial_update_ms: int = Field(
        default=250,
        description="How often (ms) to emit partial transcript updates.",
    )

    # ── Claude Code terminal injection (browser-free runner) ─────────
    cc_window_title: str = Field(
        default="",
        description=(
            "Substring of the window title to inject the transcript into "
            "(e.g. 'claude' or the terminal tab title running Claude Code). "
            "Empty = type into whatever window is currently focused."
        ),
    )
    cc_auto_focus: bool = Field(
        default=True,
        description=(
            "If the target window isn't focused, find it by title and bring it "
            "to the foreground before pasting."
        ),
    )
    cc_inject_press_enter: bool = Field(
        default=True,
        description="Press Enter after pasting to submit the prompt to Claude Code.",
    )

    # ── Universal push-to-talk dictation ─────────────────────────────
    ptt_hotkey: str = Field(
        default="ctrl+shift+space",
        description=(
            "Global push-to-talk hotkey (keyboard-library combo syntax). Hold "
            "to record, release to type at the cursor. Note: the laptop 'Fn' "
            "key cannot be used — it's handled in keyboard firmware and never "
            "reaches the OS, so it can't be detected by software."
        ),
    )
    ptt_output_mode: str = Field(
        default="type",
        description=(
            "How dictated text is emitted: 'type' (Unicode keystrokes, "
            "clipboard-safe, works everywhere) or 'paste' (clipboard + Ctrl+V, "
            "faster for long text)."
        ),
    )
    ptt_min_record_s: float = Field(
        default=0.3,
        description="Ignore push-to-talk clips shorter than this (accidental taps).",
    )
    ptt_append_space: bool = Field(
        default=True,
        description="Append a trailing space after each dictation.",
    )

    # ── Server ───────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Bind host.")
    port: int = Field(default=8000, description="Bind port.")

    model_config = {"env_prefix": "", "case_sensitive": False}


# Singleton settings instance
settings = Settings()
