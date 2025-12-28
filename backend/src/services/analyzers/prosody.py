import numpy as np
import librosa
import parselmouth
from typing import Optional
from src.schemas.audio_metrics import AudioFeatures
from src.services.analyzers.interpreter import ProsodyInterpreter

class ProsodyAnalyzer:
    def __init__(self, sample_rate: int = 16000):
        self.target_sr = sample_rate
        self.window_size = 1.5
        self.overlap_duration = 0.5
        self._buffer = np.array([], dtype=np.float32)
        self.processed_samples = 0

        self.min_voiced_prob = 0.7
        self.tonality_threshold = 0.8
        self.stability_threshold_hz = 15.0
        
        self.silence_threshold = 0.03
        
        self.interpreter = ProsodyInterpreter()

    def process(self, new_chunk: np.ndarray) -> Optional[AudioFeatures]:
        self._buffer = np.concatenate((self._buffer, new_chunk))
        self.processed_samples += len(new_chunk)

        if (len(self._buffer) / self.target_sr) < self.window_size:
            return None

        window_start = (self.processed_samples - len(self._buffer)) / self.target_sr
        window_end = self.processed_samples / self.target_sr
        metrics = self._analyze(self._buffer, window_start, window_end)

        overlap_samples = int(self.overlap_duration * self.target_sr)
        self._buffer = self._buffer[-overlap_samples:]

        return metrics

    def _analyze(self, y: np.ndarray, start_time: float, end_time: float) -> AudioFeatures:
        volume = float(np.sqrt(np.mean(y**2)))
        
        hop_length = int(self.target_sr * 0.05) 
        frame_rms = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]
        
        silent_frames = np.sum(frame_rms < self.silence_threshold)
        total_frames = len(frame_rms)
        duration = max(end_time - start_time, 1e-6)
        
        silence_ratio = float(silent_frames) / float(total_frames) if total_frames > 0 else 0.0

        sound = parselmouth.Sound(y, sampling_frequency=self.target_sr)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        pitch_values = pitch.selected_array["frequency"]

        voiced_mask = pitch_values > 0
        voiced_frequencies = pitch_values[voiced_mask]
        voiced_prob_val = float(np.mean(voiced_mask)) if len(voiced_mask) > 0 else 0.0

        if len(voiced_frequencies) > 5:
            p_mean = float(np.mean(voiced_frequencies))
            q75, q25 = np.percentile(voiced_frequencies, [75, 25])
            iqr = q75 - q25
            p_std = float(iqr / 1.35) 
        else:
            p_mean = 0.0
            p_std = 0.0

        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        tonality = 1.0 - flatness

        hesitation_val = 0.0

        if silence_ratio > 0.7:
             hesitation_val = (silence_ratio - 0.7) * 3.0
             hesitation_val = min(hesitation_val, 1.0)
             
        elif (
            tonality > self.tonality_threshold
            and voiced_prob_val > self.min_voiced_prob
            and p_std < self.stability_threshold_hz
            and p_std > 0
        ):
            stability_score = 1.0 - (p_std / self.stability_threshold_hz)
            hesitation_val = voiced_prob_val * stability_score

        features = AudioFeatures(
            start_time=start_time,
            end_time=end_time,
            pitch_mean=p_mean,
            pitch_std=p_std,
            voiced_prob=voiced_prob_val,
            hesitation_rate=max(0.0, hesitation_val),
            volume=volume,
        )
        
        features.feedback = self.interpreter.interpret(features)
        
        return features

