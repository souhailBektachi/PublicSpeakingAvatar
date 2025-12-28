from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict


class TimeInterval(BaseModel):
    start_time: float = Field(..., description="Start time of the audio chunk")
    end_time: float = Field(..., description="End time of the audio chunk")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class TimestampsSegment(TimeInterval):
    text: str
    is_final: bool = False


class FeedbackItem(BaseModel):
    status: str = Field(..., description="Status label for the metric")
    message: str = Field(..., description="Detailed feedback message")
    severity: Literal["success", "info", "warning", "error"] = Field(
        ..., description="Severity level of the feedback"
    )
    message_id: int = Field(0, description="Index of the message variant used")


class VolumeFeedbackItem(FeedbackItem):
    db_fs: float = Field(..., description="Volume in dBFS")


class ProsodyFeedback(BaseModel):
    pitch_dynamics: FeedbackItem = Field(..., description="Pitch variation analysis")
    pitch_range: FeedbackItem = Field(..., description="Vocal range assessment")
    fluency: FeedbackItem = Field(..., description="Speech fluency analysis")
    hesitation: FeedbackItem = Field(..., description="Hesitation detection")
    volume: VolumeFeedbackItem = Field(..., description="Volume level analysis")
    overall_score: float = Field(..., description="Overall speaking score (0-100)", ge=0, le=100)
    suggestions: List[str] = Field(default_factory=list, description="Actionable improvement suggestions")


class AudioFeatures(TimeInterval):

    pitch_mean: float = Field(..., description="Mean pitch of the audio in Hz")
    pitch_std: float = Field(..., description="Standard deviation of the pitch in Hz")

    voiced_prob: float = Field(
        ..., description="Probability of voiced segments in the audio"
    )
    hesitation_rate: float = Field(..., description="Rate of hesitations in the audio")

    volume: float = Field(..., description="RMS Energy (Loudness)")
    
    feedback: Optional[ProsodyFeedback] = Field(None, description="Interpreted coaching feedback")

class SummarizedInterval(TimeInterval):
    status: str
    message: str
    avg_pitch_mean: float
    avg_pitch_std: float
    avg_voiced_prob: float
    avg_volume: float
    chunk_count: int

class WarningEvent(BaseModel):
    timestamp: float
    category: str
    status: str
    message: str
    db_fs: Optional[float] = None

class FeedbackSummary(BaseModel):
    total_duration: float
    chunk_count: int
    overall_score: float
    success_intervals: List[SummarizedInterval] = []
    info_intervals: List[SummarizedInterval] = []
    warnings: List[WarningEvent] = []
    suggestions: List[str] = []
