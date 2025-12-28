import base64
import logging
from typing import Optional
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from src.core.config import settings

logger = logging.getLogger(__name__)


class ElevenLabsSynthesizer:
    """Text-to-speech synthesizer using ElevenLabs API."""

    def __init__(self, api_key: Optional[str] = None, voice_id: Optional[str] = None):
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.voice_id = voice_id or settings.ELEVENLABS_VOICE_ID

        if not self.api_key:
            logger.warning("ElevenLabs API key not found")
            self._enabled = False
            self.client = None
        else:
            self._enabled = True
            self.client = ElevenLabs(api_key=self.api_key)
            logger.info(f"ElevenLabs initialized with voice ID: {self.voice_id}")

        # Voice configuration
        self.voice_settings = VoiceSettings(
            stability=0.75,
            similarity_boost=0.75,
            style=0.25,
            use_speaker_boost=True,
        )

    def synthesize(self, text: str, voice_id: Optional[str] = None) -> Optional[str]:
        """Convert text to speech and return base64 audio."""
        if not self._enabled or not text.strip():
            return None

        try:
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id or self.voice_id,
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text.strip(),
                model_id="eleven_multilingual_v2",
                voice_settings=self.voice_settings,
            )

            audio_data = b"".join(audio_generator)
            if not audio_data:
                return None

            return base64.b64encode(audio_data).decode("utf-8")

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None

    def synthesize_to_file(self, text: str, output_path: str, voice_id: Optional[str] = None) -> bool:
        """Convert text to speech and save it to a file."""
        if not self._enabled or not text.strip():
            return False

        try:
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id or self.voice_id,
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text.strip(),
                model_id="eleven_multilingual_v2",
                voice_settings=self.voice_settings,
            )

            audio_data = b"".join(audio_generator)
            with open(output_path, "wb") as f:
                f.write(audio_data)

            logger.info(f"Audio saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"File synthesis failed: {e}")
            return False

    def get_available_voices(self) -> Optional[list]:
        """Return available ElevenLabs voices."""
        if not self._enabled:
            return None

        try:
            response = self.client.voices.get_all()
            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": v.category,
                    "description": getattr(v, "description", ""),
                }
                for v in response.voices
            ]

        except Exception as e:
            logger.error(f"Failed to retrieve voices: {e}")
            return None

    def set_voice(self, voice_id: str) -> bool:
        """Set a new default voice."""
        if not self._enabled:
            return False

        if self.synthesize("Test", voice_id):
            self.voice_id = voice_id
            logger.info(f"Voice set to {voice_id}")
            return True

        logger.error(f"Invalid voice ID: {voice_id}")
        return False

    def is_enabled(self) -> bool:
        """Check if synthesizer is enabled."""
        return self._enabled

    def get_character_limit(self) -> int:
        """Return synthesis character limit."""
        return 2500

    def split_long_text(self, text: str, max_length: Optional[int] = None) -> list:
        """Split text into chunks within character limits."""
        max_len = max_length or self.get_character_limit()
        if len(text) <= max_len:
            return [text]

        chunks, current = [], ""
        sentences = text.replace("!", ".").replace("?", ".").split(".")

        for sentence in map(str.strip, sentences):
            if not sentence:
                continue

            if len(current) + len(sentence) + 2 <= max_len:
                current = f"{current}. {sentence}" if current else sentence
            else:
                chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks
