"""
Warmup script — pre-downloads and loads the Whisper model.

Run this once after install to avoid the download penalty
on the first user utterance:

    python scripts/warmup_model.py

This script loads the same model configured in .env / environment.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from faster_whisper import WhisperModel


def main():
    model_name = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    print(f"Loading Whisper model '{model_name}' on {device} ({compute_type})...")
    start = time.time()

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    print("Warmup complete — model is cached for future runs.")


if __name__ == "__main__":
    main()
