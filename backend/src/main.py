from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import websocket
from src.core.logging import setup_logging

# Setup logging before app startup
setup_logging()

from contextlib import asynccontextmanager
import asyncio
import numpy as np
from src.services.parakeet_asr import get_asr_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
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