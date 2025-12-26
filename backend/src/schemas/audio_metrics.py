from pydantic import BaseModel, Field
from typing import Optional



class AudioFeatures(BaseModel):
    timestamp: Optional[float] = Field(None, description="Timestamp of the audio chunk")
    transcript: Optional[str] = Field(None, description="Transcript of the audio chunk")
    
    pitch_mean : float = Field(..., description="Mean pitch of the audio in Hz")
    pitch_std : float = Field(..., description="Standard deviation of the pitch in Hz")

    voiced_prob : float = Field(..., description="Probability of voiced segments in the audio")
    hesitation_rate : float = Field(..., description="Rate of hesitations in the audio")

    volume: float = Field(..., description="RMS Energy (Loudness)")