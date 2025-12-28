from pydantic import BaseModel, Field
from typing import Optional, Literal, Union


class StreamPayload(BaseModel):
    timestamp: float = Field(..., description="Timestamp of the audio chunk")
    audio_chunk: str = Field(..., description="Base64 encoded audio chunk")


class EndSessionPayload(BaseModel):
    type: Literal["end_session"] = "end_session"


class FeedbackResponse(BaseModel):
    processed_at: Optional[float] = None
    audio: Optional[str] = None
    text: Optional[str] = None
    emotion: Optional[str] = None

 
class ReportResponse(BaseModel):
    type: Literal["final_report"] = "final_report"
    report: Union[dict, list]
