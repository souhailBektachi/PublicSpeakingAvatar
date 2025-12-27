import base64
import io
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple
from src.schemas.audio_metrics import AudioFeatures, TimestampsSegment
from src.services.analyzers.prosody import ProsodyAnalyzer
from src.services.analyzers.transcriber import Transcriber

class AudioEngine:
    def __init__(self):
        self.target_sr = 16000
        self.processed_samples = 0
        self.prosody = ProsodyAnalyzer(sample_rate=self.target_sr)
        self.transcriber = Transcriber(sample_rate=self.target_sr)

    def process_stream(self, base64_chunk: str) -> Tuple[Optional[AudioFeatures], Optional[TimestampsSegment]]:
        new_audio = self._decode_chunk(base64_chunk)
        
        prosody_metrics = self.prosody.process(new_audio)
        transcript_segment = self.transcriber.process(new_audio)

        return prosody_metrics, transcript_segment

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