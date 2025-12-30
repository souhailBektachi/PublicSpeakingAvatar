"""
Lightweight real-time audio metrics analyzer.
Runs on every incoming chunk (no buffering) for responsive chart updates.
"""
import numpy as np
import parselmouth
from dataclasses import dataclass
from typing import Optional

@dataclass
class RealtimeMetrics:
    """Simple metrics computed instantly per chunk."""
    pitch: float  # Hz, 0 if unvoiced
    volume: float  # RMS (0-1 scale)
    is_voiced: bool

class RealtimeAnalyzer:
    """Fast pitch/volume analyzer for real-time chart updates."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.min_pitch = 75   # Hz
        self.max_pitch = 300  # Hz
        self.silence_threshold = 0.01
    
    def analyze(self, audio: np.ndarray) -> Optional[RealtimeMetrics]:
        """Analyze a short audio chunk and return metrics immediately."""
        if len(audio) < self.sr * 0.1:  # Need at least 100ms
            return None
        
        # Volume (RMS)
        volume = float(np.sqrt(np.mean(audio**2)))
        
        # Skip pitch if too quiet
        if volume < self.silence_threshold:
            return RealtimeMetrics(pitch=0.0, volume=volume, is_voiced=False)
        
        # Pitch via Parselmouth (fast for short clips)
        try:
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch_obj = sound.to_pitch(
                time_step=0.05,
                pitch_floor=self.min_pitch,
                pitch_ceiling=self.max_pitch
            )
            pitch_values = pitch_obj.selected_array['frequency']
            voiced = pitch_values[pitch_values > 0]
            
            if len(voiced) > 0:
                pitch = float(np.median(voiced))
                return RealtimeMetrics(pitch=pitch, volume=volume, is_voiced=True)
            else:
                return RealtimeMetrics(pitch=0.0, volume=volume, is_voiced=False)
        except Exception:
            return RealtimeMetrics(pitch=0.0, volume=volume, is_voiced=False)
