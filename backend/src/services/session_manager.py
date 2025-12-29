from fastapi import WebSocket
from typing import Dict, Optional, List, Generator, Tuple
import logging
import time
from src.services.audio_engine import AudioEngine
from src.services.live_analyzer import LiveAnalyzer
from src.services.llm_coach import LLMCoach
from src.services.analyzers.summarizer import FeedbackSummarizer
from src.services.synthesizers.elevenlabs import ElevenLabsSynthesizer
from src.schemas.audio_metrics import AudioFeatures, TimestampsSegment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a seasoned Public Speaking Coach giving live, meaningful feedback. "
    "RULES:\n"
    "1. Format: [Emotion] Your feedback\n"
    "2. Keep it SHORT (3-8 words)\n"
    "3. Give SPECIFIC, ACTIONABLE praise or tips\n"
    "4. Reference what they ACTUALLY said\n"
    "5. NO generic phrases like 'Tell me more', 'Keep going', 'Great job'\n"
    "6. Emotions: Impressed, Encouraging, Insightful, Confident, Supportive\n\n"
    "Good Examples:\n"
    "[Impressed] Vivid rocket analogy!\n"
    "[Encouraging] Strong statistic, cite the source!\n"
    "[Confident] Your pace is perfect here.\n"
    "[Insightful] That comparison lands well.\n"
    "[Supportive] Pause after that point.\n\n"
    "Bad Examples (NEVER use):\n"
    "- 'Tell me more' (vague)\n"
    "- 'Great point' (generic)\n"
    "- 'Keep going' (empty)"
)


class AudioCoordinator:
    """Coordinates audio output with round-robin alternation between sources."""
    
    def __init__(self, cooldown: float = 5.0):
        self.cooldown = cooldown
        self.last_audio_time: float = 0.0
        self.last_source: str = None  # Track who sent last for alternation
    
    def can_send_audio(self, source: str) -> bool:
        """Check if this source can send audio (cooldown + alternation)."""
        time_ok = time.time() - self.last_audio_time >= self.cooldown
        # Alternate: different source OR first time
        turn_ok = self.last_source is None or self.last_source != source
        return time_ok and turn_ok
    
    def mark_audio_sent(self, source: str):
        """Mark that an audio was just sent by this source."""
        self.last_audio_time = time.time()
        self.last_source = source
    
    def time_until_available(self) -> float:
        """Returns seconds until next audio can be sent."""
        remaining = self.cooldown - (time.time() - self.last_audio_time)
        return max(0.0, remaining)


class Session:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.engine = AudioEngine() 
        self.live_analyzer = LiveAnalyzer()
        self.llm_coach = LLMCoach()
        self.summarizer = FeedbackSummarizer()
        self.tts = ElevenLabsSynthesizer()
        self.audio_coordinator = AudioCoordinator(cooldown=5.0)
        
        # State
        self.transcript_history: List[TimestampsSegment] = []
        self.metrics_history: List[AudioFeatures] = []
        self.last_feedback_time: float = 0.0
        self.last_telemetry_time: float = time.time()
        self.last_speech_end_time: float = 0.0
        self.silence_alert_sent: bool = False
        self.latest_pose: dict = {}
        self.llm_context: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def reset(self):
        """Reset session state for a new recording while keeping WebSocket alive."""
        self.engine.shutdown() # Stop transcribers
        self.engine = AudioEngine()
        self.transcript_history = []
        self.metrics_history = []
        self.last_feedback_time = 0.0
        self.last_speech_end_time = 0.0
        self.silence_alert_sent = False
        self.llm_context = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info("Session state reset (WebSocket remains open)")

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