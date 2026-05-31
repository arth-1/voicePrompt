"""
Voice-Claude Bridge — FastAPI application entry point.

Boots the server, loads the Whisper model, initializes the Claude
client, and mounts all routes including the WebSocket endpoint.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .api.health import router as health_router
from .api.websocket import voice_session_handler
from .config import settings
from .llm.base import LLMClient
from .llm.claude_client import ClaudeClient
from .llm.claude_code_client import ClaudeCodeClient
from .stt.whisper_engine import WhisperEngine
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Shared instances (initialized at startup)
_whisper: WhisperEngine | None = None
_claude: LLMClient | None = None

# Path to the frontend
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"


def _build_llm_backend() -> LLMClient:
    """Construct the LLM backend selected via settings.llm_backend."""
    backend = settings.llm_backend.strip().lower()

    if backend == "claude_code":
        workdir = settings.claude_code_workdir.strip() or str(Path.cwd())
        allowed = [
            t.strip()
            for t in settings.claude_code_allowed_tools.split(",")
            if t.strip()
        ]
        logger.info(f"Using native Claude Code backend (workdir={workdir})")
        return ClaudeCodeClient(
            working_dir=workdir,
            model=settings.claude_model,
            permission_mode=settings.claude_code_permission_mode,
            allowed_tools=allowed,
            max_turns=settings.claude_code_max_turns,
        )

    if backend != "api":
        logger.warning(
            f"Unknown LLM_BACKEND '{settings.llm_backend}', falling back to 'api'"
        )
    logger.info("Using direct Anthropic Messages API backend")
    return ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, cleanup on shutdown."""
    global _whisper, _claude

    setup_logging()
    logger.info("=" * 60)
    logger.info("  Voice-Claude Bridge starting up")
    logger.info("=" * 60)
    logger.info(f"  Whisper model : {settings.whisper_model}")
    logger.info(f"  Whisper device: {settings.whisper_device}")
    logger.info(f"  LLM backend   : {settings.llm_backend}")
    logger.info(f"  Claude model  : {settings.claude_model}")
    logger.info(f"  Sample rate   : {settings.sample_rate} Hz")
    logger.info(f"  VAD mode      : {settings.vad_mode}")
    logger.info("=" * 60)

    # Load Whisper model (blocks until downloaded/loaded)
    loop = asyncio.get_event_loop()
    _whisper = await loop.run_in_executor(
        None,
        lambda: WhisperEngine(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        ),
    )

    # Initialize the chosen LLM backend
    _claude = _build_llm_backend()

    logger.info("All services initialized — ready to accept connections")
    yield

    logger.info("Shutting down Voice-Claude Bridge")


# ── FastAPI app ─────────────────────────────────────────────────
app = FastAPI(
    title="Voice-Claude Bridge",
    description="Real-time voice-to-Claude assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
app.include_router(health_router)


# ── WebSocket endpoint ──────────────────────────────────────────
@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """WebSocket endpoint for voice sessions."""
    if _whisper is None or _claude is None:
        await websocket.close(code=1013, reason="Server not ready")
        return
    await voice_session_handler(websocket, _whisper, _claude)


# ── Serve frontend ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse(
        "<h1>Voice-Claude Bridge</h1>"
        "<p>Frontend not found. Place index.html in the frontend/ directory.</p>",
        status_code=200,
    )
