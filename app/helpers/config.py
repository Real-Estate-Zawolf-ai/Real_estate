"""Application settings loaded from environment variables and ``.env``."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration with safe defaults for local development."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Real Estate Assistant API"
    APP_VERSION: str = "0.1.0"

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GEMINI_API_KEY: str | None = None
    GEMINI_LIVE_MODEL: str = "gemini-3.1-flash-live-preview"
    GEMINI_VOICE: str = "Aoede"
    WHATSAPP_VERIFY_TOKEN: str | None = None
    WHATSAPP_APP_SECRET: str | None = None
    WHATSAPP_ACCESS_TOKEN: str | None = None
    WHATSAPP_PHONE_NUMBER_ID: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_API_URL: str | None = None
    COHERE_API_KEY: str | None = None

    GENERATION_BACKEND: str = "groq"
    GENERATION_MODEL_ID: str | None = None
    EMBEDDING_BACKEND: str | None = None
    EMBEDDING_MODEL_ID: str | None = None

    FILE_ALLOWED_TYPES: list[str] = []
    FILE_MAX_SIZE: int | None = None
    FILE_DEFAULT_CHUNK_SIZE: int | None = None

    POSTGRES_USERNAME: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_MAIN_DATABASE: str | None = None

    VECTOR_DB_BACKEND: str | None = None
    VECTOR_DB_PATH: str | None = None
    VECTOR_DB_DISTANCE_METHOD: str | None = None

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"


@lru_cache
def get_settings() -> Settings:
    """Return one settings instance for the process."""
    return Settings()
