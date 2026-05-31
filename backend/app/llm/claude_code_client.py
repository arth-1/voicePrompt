"""
Native Claude Code client.

Instead of calling the Anthropic Messages API directly, this backend drives
the **Claude Code CLI** (`claude`) in headless streaming mode. That gives the
voice assistant the full agentic Claude Code experience — it can read and edit
files and run commands inside the configured working directory — rather than a
plain chat completion.

How it works:
- Spawns ``claude -p --output-format stream-json --verbose`` as a subprocess.
- Writes the finalized transcript to the process **stdin** (never as a shell
  argument), so spoken text with quotes or shell metacharacters can't cause
  command injection.
- Reads newline-delimited JSON events from stdout and yields text as it
  arrives.

Requirements:
- The ``claude`` CLI must be installed and authenticated on the host.
  Install with:  ``npm install -g @anthropic-ai/claude-code``
  Authenticate by running ``claude`` once, or by setting ``ANTHROPIC_API_KEY``.

Docs: https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless
"""

import asyncio
import json
import shutil
from typing import AsyncGenerator, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeCodeClient:
    """
    Streaming client backed by the native Claude Code CLI.

    Args:
        working_dir: Directory Claude Code operates in (its repo context).
        model: Optional model override (e.g. 'claude-haiku-4-5'). If empty,
            the CLI uses its configured default.
        claude_bin: Path/name of the claude executable. Auto-resolved if None.
        permission_mode: CLI permission mode. Defaults to a read-only-ish
            'default'; set to 'acceptEdits' or 'bypassPermissions' to let it
            modify files autonomously.
        allowed_tools: Optional list of tools to allow (e.g. ['Read','Grep']).
        max_turns: Safety cap on agent turns per request.
    """

    def __init__(
        self,
        working_dir: str,
        model: str = "",
        claude_bin: Optional[str] = None,
        permission_mode: str = "default",
        allowed_tools: Optional[list[str]] = None,
        max_turns: int = 12,
    ):
        self.working_dir = working_dir
        self.model = model
        self.permission_mode = permission_mode
        self.allowed_tools = allowed_tools or []
        self.max_turns = max_turns
        self.claude_bin = claude_bin or self._resolve_binary()

        if self.claude_bin:
            logger.info(
                f"Claude Code client ready (bin='{self.claude_bin}', "
                f"cwd='{self.working_dir}', mode='{self.permission_mode}')"
            )
        else:
            logger.warning(
                "Claude Code CLI not found on PATH. Install with "
                "'npm install -g @anthropic-ai/claude-code'. "
                "Voice requests will report the CLI is unavailable."
            )

    @staticmethod
    def _resolve_binary() -> Optional[str]:
        """Find the claude executable on PATH (handles claude / claude.cmd)."""
        for name in ("claude", "claude.cmd", "claude.exe"):
            found = shutil.which(name)
            if found:
                return found
        return None

    def _build_command(self) -> list[str]:
        """Assemble the headless streaming CLI invocation."""
        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",  # required by the CLI when using stream-json with -p
            "--permission-mode",
            self.permission_mode,
            "--max-turns",
            str(self.max_turns),
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.allowed_tools:
            cmd += ["--allowed-tools", ",".join(self.allowed_tools)]
        return cmd

    async def stream_answer(
        self,
        user_text: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run the transcript through Claude Code and stream text deltas.

        Args:
            user_text: The finalized voice transcript (sent via stdin).
            cancel_event: If set mid-stream, the subprocess is terminated
                (barge-in support).

        Yields:
            Text delta strings as Claude Code produces them.
        """
        if not self.claude_bin:
            yield "[Claude Code unavailable: 'claude' CLI not installed. Run: npm install -g @anthropic-ai/claude-code]"
            return

        cmd = self._build_command()
        logger.info(f"Launching Claude Code: '{user_text[:80]}'")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.working_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"Failed to launch Claude Code CLI: {e}")
            yield f"[Claude Code launch failed: {e}]"
            return

        # Send the transcript via stdin, then close it so the CLI starts work.
        try:
            assert proc.stdin is not None
            proc.stdin.write(user_text.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except Exception as e:
            logger.error(f"Failed writing to Claude Code stdin: {e}")

        emitted_any = False
        try:
            async for delta in self._read_stream(proc, cancel_event):
                emitted_any = True
                yield delta
        finally:
            await self._cleanup(proc, cancel_event)

        if not emitted_any:
            stderr = b""
            try:
                if proc.stderr is not None:
                    stderr = await proc.stderr.read()
            except Exception:
                pass
            detail = stderr.decode("utf-8", "replace").strip()
            if detail:
                logger.error(f"Claude Code produced no output. stderr: {detail[:500]}")
                yield f"[Claude Code error: {detail[:300]}]"

    async def _read_stream(
        self,
        proc: asyncio.subprocess.Process,
        cancel_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[str, None]:
        """Parse newline-delimited JSON events from the CLI stdout."""
        assert proc.stdout is not None
        first_text = True

        while True:
            if cancel_event and cancel_event.is_set():
                logger.info("Claude Code stream cancelled by user interruption")
                return

            try:
                line = await proc.stdout.readline()
            except Exception as e:
                logger.error(f"Error reading Claude Code stdout: {e}")
                return

            if not line:
                break  # EOF

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON line (rare); skip it.
                continue

            text = self._extract_text(event)
            if text:
                if first_text:
                    logger.info("Claude Code first token received")
                    first_text = False
                yield text

    @staticmethod
    def _extract_text(event: dict) -> str:
        """
        Pull human-facing text out of a Claude Code stream-json event.

        Handles two shapes:
        - partial streaming deltas:
            {"type":"stream_event","event":{"type":"content_block_delta",
             "delta":{"type":"text_delta","text":"..."}}}
        - complete assistant messages:
            {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}

        The final {"type":"result","result":"..."} event is intentionally
        ignored here to avoid duplicating already-streamed text.
        """
        etype = event.get("type")

        # Partial token deltas (preferred — gives the live typewriter effect).
        if etype == "stream_event":
            inner = event.get("event", {})
            if inner.get("type") == "content_block_delta":
                delta = inner.get("delta", {})
                if delta.get("type") == "text_delta":
                    return delta.get("text", "")
            return ""

        # Whole assistant message (when partial streaming isn't available).
        if etype == "assistant":
            msg = event.get("message", {})
            parts = msg.get("content", [])
            if isinstance(parts, list):
                return "".join(
                    p.get("text", "")
                    for p in parts
                    if isinstance(p, dict) and p.get("type") == "text"
                )
        return ""

    async def _cleanup(
        self,
        proc: asyncio.subprocess.Process,
        cancel_event: Optional[asyncio.Event],
    ) -> None:
        """Terminate the subprocess if still running and reap it."""
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
