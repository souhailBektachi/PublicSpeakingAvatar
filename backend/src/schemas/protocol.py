from pydantic import BaseModel, Field
from typing import Optional, Literal, Union


class StreamPayload(BaseModel):
    timestamp: float = Field(..., description="Timestamp of the audio chunk")
    audio_chunk: str = Field(..., description="Base64 encoded audio chunk")


class EndSessionPayload(BaseModel):
    type: Literal["end_session"] = "end_session"


class FeedbackResponse(BaseModel):
    type: str = "feedback"
    feedback: Optional[str] = None
    audio: Optional[str] = None
    visemes: Optional[dict] = None  # Lip sync data: {duration, visemes: [{time, shape}]}
    sentiment: Optional[str] = None
    timestamp: Optional[float] = None

 
class ReportResponse(BaseModel):
    type: Literal["final_report"] = "final_report"
    report: Union[dict, list]


class TelemetryPayload(BaseModel):
    type: Literal["telemetry"]
    timestamp: float
    pose_data: dict
    is_speaking: bool
