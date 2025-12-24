from pydantic import BaseModel, Field



class AudioFeatures(BaseModel):
    pitch_mean : float = Field(..., description="Mean pitch of the audio in Hz")
    pitch_std : float = Field(..., description="Standard deviation of the pitch in Hz")

    voiced_prob : float = Field(..., description="Probability of voiced segments in the audio")
    hesitation_rate : float = Field(..., description="Rate of hesitations in the audio")

    volume: float = Field(..., description="RMS Energy (Loudness)")