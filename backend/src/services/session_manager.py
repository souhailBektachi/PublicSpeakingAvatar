from fastapi import WebSocket
from typing import Dict, Optional
import logging
from src.services.audio_engine import AudioEngine
from src.services.live_analyzer import LiveAnalyzer
from src.schemas.audio_metrics import AudioFeatures, TimestampsSegment

logger = logging.getLogger(__name__)

class Session:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.engine = AudioEngine() 
        self.live_analyzer = LiveAnalyzer()
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

    async def store_results(
        self,
        session_id: str,
        metrics: Optional[AudioFeatures],
        transcript: Optional[TimestampsSegment]
    ):
        if session_id not in self.active_sessions:
            return
        session = self.active_sessions[session_id]
        
        if metrics:
            session.metrics_history.append(metrics)
            
        if transcript and transcript.text:
            session.transcript_history.append(transcript.text)
    
    def get_recent_metrics(self , session_id: str , limit : int = 10) -> list[AudioFeatures]:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].metrics_history[-limit:]
        return []
    def get_full_transcript(self, session_id: str) -> str:
        if session_id in self.active_sessions:
            return " ".join(self.active_sessions[session_id].transcript_history)
        return ""

manager = SessionManager()