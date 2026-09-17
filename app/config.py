from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Real Estate Assistant API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # OCR Configuration
    OCR_PROVIDER: str = "paddleocr"
    OCR_LANGUAGES: list[str] = ["ar", "en"]
    OCR_GPU: bool = False
    OCR_MIN_CONFIDENCE: float = 0.30
    OCR_MODEL_DIR: str | None = None
    OCR_RENDER_DPI: int = 300
    OCR_MIN_TEXT_CHARS_PER_PAGE: int = 40

    # API Keys & Services
    GROQ_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # Storage & DB
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_DB: str | None = None
    QDRANT_URL: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()

