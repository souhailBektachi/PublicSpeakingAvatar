from fastapi import WebSocket
from typing import Dict, Optional, List, Generator, Tuple
import logging
from src.services.audio_engine import AudioEngine
from src.services.live_analyzer import LiveAnalyzer
from src.services.llm_coach import LLMCoach
from src.services.analyzers.summarizer import FeedbackSummarizer
from src.schemas.audio_metrics import AudioFeatures, TimestampsSegment

logger = logging.getLogger(__name__)

class Session:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.engine = AudioEngine() 
        self.live_analyzer = LiveAnalyzer()
        self.llm_coach = LLMCoach()
        self.summarizer = FeedbackSummarizer()
        
        # State
        self.transcript_history: List[TimestampsSegment] = []
        self.metrics_history: List[AudioFeatures] = []
        self.last_feedback_time: float = 0.0
        
        # LLM Context (Conversation History)
        self.llm_context: List[Dict[str, str]] = [
            {
                "role": "system", 
                "content": (
                    "You are a charismatic Public Speaking Coach giving live feedback. "
                    "RULES:\n"
                    "1. Output EXACTLY this format: [Emotion] Your message\n"
                    "2. Keep messages SHORT (1-5 words max)\n"
                    "3. Focus on ENCOURAGING the speaker\n"
                    "4. DO NOT explain, analyze, or describe what they said\n"
                    "5. DO NOT start with 'Okay', 'So', 'The user'\n"
                    "6. Emotions: Excited, Curious, Impressed, Encouraging, Supportive\n\n"
                    "Examples:\n"
                    "[Excited] Great analogy!\n"
                    "[Impressed] Powerful opening!\n"
                    "[Curious] Tell me more!\n"
                    "[Encouraging] Keep going!"
                )
            }
        ]

    async def send_json(self, data: dict):
        await self.websocket.send_json(data)

    def generate_report(self) -> dict:
        """
        Generates final report using accumulated history.
        """
        summary = self.summarizer.summarize(self.metrics_history)
        return self.llm_coach.generate_final_report(self.transcript_history, summary, self.llm_context)

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
            
        # Store full transcript segment
        if transcript and transcript.text and getattr(transcript, "is_final", False):
            session.transcript_history.append(transcript)
    
    def get_recent_metrics(self , session_id: str , limit : int = 10) -> list[AudioFeatures]:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].metrics_history[-limit:]
        return []

    def get_full_transcript(self, session_id: str) -> str:
        if session_id in self.active_sessions:
            return " ".join([t.text for t in self.active_sessions[session_id].transcript_history])
        return ""

manager = SessionManager()