import numpy as np
import librosa
import crepe
from typing import Optional
from src.schemas.audio_metrics import AudioFeatures

class ProsodyAnalyzer:
    def __init__(self, sample_rate: int = 16000):
        self.target_sr = sample_rate
        self.window_size = 1.5
        self.overlap_duration = 0.5
        self._buffer = np.array([], dtype=np.float32)
        self.confidence_threshold = 0.5

    def process(self, new_chunk: np.ndarray) -> Optional[AudioFeatures]:
        self._buffer = np.concatenate((self._buffer, new_chunk))

        if (len(self._buffer) / self.target_sr) < self.window_size:
            return None

        metrics = self._analyze(self._buffer)

        overlap_samples = int(self.overlap_duration * self.target_sr)
        self._buffer = self._buffer[-overlap_samples:]

        return metrics

    def _analyze(self, y: np.ndarray) -> AudioFeatures:
        volume = float(np.sqrt(np.mean(y ** 2)))

        _, frequency, confidence, _ = crepe.predict(
            y,
            self.target_sr,
            model_capacity="small",
            step_size=10,
            viterbi=False,
            verbose=0
        )

        voiced_mask = confidence > self.confidence_threshold
        voiced_frequencies = frequency[voiced_mask]

        if len(voiced_frequencies) > 0:
            p_mean = float(np.mean(voiced_frequencies))
            p_std = float(np.std(voiced_frequencies))
        else:
            p_mean = 0.0
            p_std = 0.0

        voiced_prob_val = float(np.mean(voiced_mask))

        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        tonality = 1.0 - flatness

        hesitation_val = 0.0
        if tonality > 0.6 and voiced_prob_val > 0.8 and p_std < 5.0:
            hesitation_val = float(tonality * voiced_prob_val * (1.0 - min(p_std / 10.0, 1.0)))

        return AudioFeatures(
            pitch_mean=p_mean,
            pitch_std=p_std,
            voiced_prob=voiced_prob_val,
            hesitation_rate=hesitation_val,
            volume=volume
        )