import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "AEGIS - AI Police Assistance System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "a_very_secret_key_for_development_only_12345"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30 # 30 days
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/aegis")
    RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8080")

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()
