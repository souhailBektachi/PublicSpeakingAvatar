import base64
import io
import numpy as np
import soundfile as sf
import librosa
from typing import Optional
from src.schemas.audio_metrics import AudioFeatures
from src.services.analyzers.prosody import ProsodyAnalyzer
from src.services.analyzers.transcriber import Transcriber

class AudioEngine:
    def __init__(self):
        self.target_sr = 16000
        self.processed_samples = 0
        self.prosody = ProsodyAnalyzer(sample_rate=self.target_sr)
        self.transcriber = Transcriber(sample_rate=self.target_sr)

    def process_stream(self, base64_chunk: str) -> Optional[AudioFeatures]:
        new_audio = self._decode_chunk(base64_chunk)
        
        self.processed_samples += len(new_audio)
        current_time = self.processed_samples / self.target_sr

        prosody_metrics = self.prosody.process(new_audio)
        transcript_text = self.transcriber.process(new_audio)

        if prosody_metrics:
            prosody_metrics.timestamp = current_time
            if transcript_text:
                prosody_metrics.transcript = transcript_text
            return prosody_metrics
        
        if transcript_text:
            return AudioFeatures(
                timestamp=current_time,
                pitch_mean=0.0,
                pitch_std=0.0,
                voiced_prob=0.0,
                hesitation_rate=0.0,
                volume=0.0,
                transcript=transcript_text
            )

        return None

    def shutdown(self):
        self.transcriber.stop()

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