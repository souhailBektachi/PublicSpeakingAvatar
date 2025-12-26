import numpy as np
import librosa
from typing import Optional
from src.schemas.audio_metrics import AudioFeatures

class ProsodyAnalyzer:
    def __init__(self, sample_rate: int = 16000):
        self.target_sr = sample_rate
        self.window_size = 1.5
        self._buffer = np.array([], dtype=np.float32)

    def process(self, new_chunk: np.ndarray) -> Optional[AudioFeatures]:
        self._buffer = np.concatenate((self._buffer, new_chunk))

        if (len(self._buffer) / self.target_sr) < self.window_size:
            return None

        metrics = self._analyze(self._buffer)

        overlap_samples = int(0.5 * self.target_sr)
        self._buffer = self._buffer[-overlap_samples:]

        return metrics

    def _analyze(self, y: np.ndarray) -> AudioFeatures:
        rms = float(np.mean(librosa.feature.rms(y=y)))

        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=300, sr=self.target_sr)
        
        speech_pitch = f0[voiced_flag]
        if len(speech_pitch) > 0:
            p_mean = float(np.mean(speech_pitch))
            p_std = float(np.std(speech_pitch))
        else:
            p_mean = 0.0
            p_std = 0.0

        voiced_prob_val = float(np.mean(voiced_flag))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        tonality = 1.0 - flatness 
        
        hesitation_val = 0.0
        if tonality > 0.6 and voiced_prob_val > 0.8 and p_std < 5.0:
            hesitation_val = float(tonality)

        return AudioFeatures(
            pitch_mean=p_mean,
            pitch_std=p_std,
            voiced_prob=voiced_prob_val,
            hesitation_rate=hesitation_val,
            volume=rms
        )