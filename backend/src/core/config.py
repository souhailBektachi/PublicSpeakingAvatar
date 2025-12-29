import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AssemblyAI_API_KEY: str
    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str = "pqHfZKP75CvOlQylNhV4"  # Bill
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

print(f"[CONFIG] ELEVENLABS_API_KEY loaded: {bool(settings.ELEVENLABS_API_KEY)} (Length: {len(settings.ELEVENLABS_API_KEY)})")
if settings.ELEVENLABS_API_KEY:
    print(f"[CONFIG] Key starts with: {settings.ELEVENLABS_API_KEY[:4]}...")
