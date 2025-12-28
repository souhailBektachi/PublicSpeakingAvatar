import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AssemblyAI_API_KEY: str
    GROQ_API_KEY: str
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
