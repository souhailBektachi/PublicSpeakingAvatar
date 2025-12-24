import base64
import io
import numpy as np
import soundfile as sf
import librosa 
from typing import Optional
from src.schemas.audio_metrics import AudioFeatures
from src.services.analyzers.prosody import ProsodyAnalyzer

class AudioEngine:
    def __init__(self):
        self.target_sr = 16000
        self.analyzer = ProsodyAnalyzer(sample_rate=self.target_sr)
        self._buffer = np.array([], dtype=np.float32)

    def process_stream(self, base64_chunk: str) -> Optional[AudioFeatures]:
        new_audio = self._decode_chunk(base64_chunk)
        self._buffer = np.concatenate((self._buffer, new_audio))

        current_duration = len(self._buffer) / self.target_sr
        
        if current_duration < self.analyzer.window_size:
            return None

        metrics = self.analyzer.analyze(self._buffer)

        overlap_samples = int(0.5 * self.target_sr)
        self._buffer = self._buffer[-overlap_samples:]

        return metrics

    def _decode_chunk(self, base64_chunk: str) -> np.ndarray:
        try:
            audio_bytes = base64.b64decode(base64_chunk)
            with io.BytesIO(audio_bytes) as b:
                y, sr = sf.read(b, dtype='float32')
            
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            return y
        except Exception:
            return np.array([], dtype=np.float32)

audio_engine = AudioEngine()