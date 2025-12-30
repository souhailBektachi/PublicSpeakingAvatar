from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import websocket
from src.core.logging import setup_logging

# Setup logging before app startup
setup_logging()

from contextlib import asynccontextmanager
import asyncio
import subprocess
import sys
from pathlib import Path
import numpy as np
from src.services.parakeet_asr import get_asr_instance
from src.services.synthesizers.kokoro import get_kokoro_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rhubarb Setup: Ensure CLI is installed locally
    try:
        print("Checking Rhubarb Lip Sync installation...")
        script_path = Path(__file__).parent.parent / "scripts" / "install_rhubarb.py"
        await asyncio.to_thread(subprocess.run, [sys.executable, str(script_path)], check=True)
    except Exception as e:
        print(f"Rhubarb Auto-Install Failed: {e}")

    # Startup: Load the model and warm it up
    try:
        print("Preloading Parakeet ASR model...")
        asr = get_asr_instance()
        
        # Warmup with silence
        print("Warming up ASR (JIT/CUDA compilation)...")
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        # Run in thread pool to not block startup if it takes a few seconds
        await asyncio.to_thread(asr.transcribe, silence)
        print("Parakeet ASR Ready!")
    except Exception as e:
        print(f"ASR Preload Failed: {e}")

    # Startup: Load Kokoro TTS
    try:
        print("Preloading Kokoro TTS (Puck)...")
        # Initialize the engine (loads model + voices)
        engine = get_kokoro_engine()
        print(f"Kokoro Loaded. Default voice: {engine.voice}")
        
        # Warmup with short text
        print("Warming up Kokoro (JIT/CUDA compilation)...")
        await asyncio.to_thread(engine.generate_speech_audio, "Warmup.")
        print("Kokoro TTS Ready!")
    except Exception as e:
        print(f"Kokoro Preload Failed: {e}")

    
    yield
    # Shutdown logic (if any) could go here

app = FastAPI(title="SpeechMirror Core", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket.router)

@app.get("/")
def health_check():
    return {"status": "SpeechMirror backend is running"}