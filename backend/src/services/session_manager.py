from fastapi import WebSocket
from typing import Dict, Optional
import logging
from src.services.audio_engine import AudioEngine
from src.schemas.audio_metrics import AudioFeatures

logger = logging.getLogger(__name__)

class Session:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.engine = AudioEngine() 
        self.transcript_history: list[str] = []
        self.metrics_history: list[AudioFeatures] = []

    async def send_json(self, data: dict):
        await self.websocket.send_json(data)

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        session = Session(websocket)
        self.active_sessions[session_id] = session
        logger.info(f"Session {session_id} connected")

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].engine.shutdown()
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} disconnected")

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.active_sessions.get(session_id)

    def update_transcript(self, session_id: str, text: str):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].transcript_history.append(text)

    def update_metrics(self, session_id: str, metrics: AudioFeatures):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].metrics_history.append(metrics)
    
    def get_recent_metrics(self , session_id: str , limit : int = 10) -> list[AudioFeatures]:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].metrics_history[-limit:]
        return []
    def get_full_transcript(self, session_id: str) -> str:
        if session_id in self.active_sessions:
            return " ".join(self.active_sessions[session_id].transcript_history)
        return ""

manager = SessionManager()