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
