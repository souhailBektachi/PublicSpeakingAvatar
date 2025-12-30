import io
import os
from pathlib import Path
import base64
import asyncio
import uuid
import wave
import numpy as np
import soundfile as sf
from functools import lru_cache

from .kokoro_engine.engine import SpeechSynthesizer as KokoroEngine
from ..rhubarb_service import get_rhubarb_service

@lru_cache(maxsize=1)
def get_kokoro_engine() -> KokoroEngine:
    """Singleton getter for the Kokoro Engine."""
    return KokoroEngine()

class KokoroSynthesizer:
    def __init__(self, voice_id: str = "am_michael"):
        """
        Initialize the Kokoro synthesizer.
        Args:
            voice_id: The specific voice ID to use (default: 'am_michael').
        """
        self.engine = get_kokoro_engine()
        self.voice_id = voice_id
        self.engine.set_voice(voice_id)

    def synthesize(self, text: str) -> str:
        """Synthesize text to audio (base64-encoded WAV)."""
        # Run synchronous engine code directly
        audio_float = self.engine.generate_speech_audio(text)
        
        # Convert float32 numpy array to 16-bit PCM bytes (WAV format)
        # 24000 Hz is Kokoro default
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_float, 24000, format='WAV', subtype='PCM_16')
            return base64.b64encode(wav_buffer.getvalue()).decode('utf-8')

    def is_enabled(self) -> bool:
        """Check if synthesizer is enabled (always true for local)."""
        return True

    def synthesize_with_visemes(self, text: str):
        """
        Synthesize text and return (audio_base64, visemes).
        """
        # 1. Generate Audio
        audio_float = self.engine.generate_speech_audio(text)
        
        # 2. Save to temporary WAV file for Rhubarb
        temp_filename = f"temp_tts_{uuid.uuid4().hex}.wav"
        temp_path = Path("temp") / temp_filename
        Path("temp").mkdir(exist_ok=True)
        
        sf.write(str(temp_path), audio_float, 24000)
        
        try:
            # 3. Get Visemes via RhubarbService
            rhubarb = get_rhubarb_service()
            lip_sync_result = rhubarb.extract_visemes(
                base64.b64encode(open(temp_path, "rb").read()).decode('utf-8'), 
                audio_format="wav"
            )
            
            # 4. Read back audio as bytes for base64 return
            # (We already have it in temp_path, but simpler to just encode from bytes we'd write)
            with open(temp_path, "rb") as f:
                wav_bytes = f.read()
                
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            visemes = None
            if lip_sync_result:
                visemes = rhubarb.visemes_to_dict(lip_sync_result)

            return {
                "audio": audio_base64,
                "visemes": visemes
            }
            
        finally:
            # Cleanup temp file
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except:
                    pass
