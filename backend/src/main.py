from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import websocket
from src.core.logging import setup_logging

# Setup logging before app startup
setup_logging()

app = FastAPI(title="SpeechMirror Core")

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