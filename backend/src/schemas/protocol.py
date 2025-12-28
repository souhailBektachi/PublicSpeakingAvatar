from pydantic import BaseModel, Field
from typing import Optional




class StreamPayload(BaseModel):
    timestamp: float = Field(..., description="Timestamp of the audio chunk")
    audio_chunk: str = Field(..., description="Base64 encoded audio chunk")


class FeedbackResponse(BaseModel):
    processed_at: float
    audio: Optional[str] = None
