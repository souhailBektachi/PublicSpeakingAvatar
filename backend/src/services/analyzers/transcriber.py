import logging
import time
import numpy as np
from typing import Optional, List

from src.schemas.audio_metrics import TimestampsSegment
from src.services.parakeet_asr import ParakeetASR, get_asr_instance

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.processed_samples = 0
        
        try:
            logger.info("Getting Parakeet ASR instance...")
            self.asr = get_asr_instance()
            self._enabled = True
            logger.info("Parakeet ASR initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Parakeet ASR: {e}")
            logger.warning("Transcription will be disabled.")
            self._enabled = False

    def process(self, new_chunk: np.ndarray, timestamp: float = None) -> Optional[TimestampsSegment]:
        """
        Process a new audio chunk (float32 numpy array).
        Since the frontend sends 'thoughts' (complete sentences), 
        we treat each chunk as a complete utterance to transcribe.
        """
        if len(new_chunk) == 0 or not self._enabled:
            return None
        
        # Calculate timing
        duration = len(new_chunk) / self.sample_rate
        
        if timestamp is not None:
            end_time = timestamp
            start_time = end_time - duration
        else:
            start_time = self.processed_samples / self.sample_rate
            end_time = start_time + duration
        self.processed_samples += len(new_chunk)
        
        text = ""
        try:
             # Synchronous transcription
            text = self.asr.transcribe(new_chunk)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

        if not text:
            return None

        logger.info(f"Parakeet Transcript: '{text}'")

        return TimestampsSegment(
            text=text,
            start_time=start_time,
            end_time=end_time,
            is_final=True
        )

    def get_result(self) -> Optional[TimestampsSegment]:
        """
        Legacy method compatibility. 
        Since we process synchronously now, this can return None or be removed if unused.
        AudioEngine calls it for 'async transcripts' but we return everything in process().
        """
        return None

    def flush(self, timeout: float = 5.0) -> List[TimestampsSegment]:
        """
        No-op for synchronous local ASR.
        """
        return []

    def stop(self):
        pass