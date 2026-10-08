"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with safe defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    APP_NAME: str = "Real Estate Assistant API"
    APP_VERSION: str = "0.1.0"
    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    # Comma-separated browser origins allowed for CORS and the voice
    # WebSocket. "*" keeps local development open; set real origins in prod.
    CORS_ALLOWED_ORIGINS: str = "*"

    # HMAC key for website session ids and voice tokens. When unset a random
    # per-process key is used, so sessions do not survive a restart.
    SESSION_SECRET: str | None = None

    # Required (X-Admin-Key header) for endpoints that write to the property
    # index: POST /properties and POST /documents/ingest. Unset = disabled.
    ADMIN_API_KEY: str | None = None

    # -----------------------------------------------------------------------
    # Voice (website, Gemini Live)
    # -----------------------------------------------------------------------

    VOICE_TOKEN_TTL_SECONDS: int = 60
    VOICE_MAX_SESSION_SECONDS: int = 600
    VOICE_IDLE_TIMEOUT_SECONDS: int = 90
    VOICE_MAX_CONCURRENT_SESSIONS: int = 20
    VOICE_MAX_FRAME_BYTES: int = 64_000

    # -----------------------------------------------------------------------
    # Scheduling
    # -----------------------------------------------------------------------

    VIEWING_TIMEZONE: str = "Africa/Cairo"
    VIEWING_DAY_START_HOUR: int = 10
    VIEWING_DAY_END_HOUR: int = 18
    VIEWING_SLOT_MINUTES: int = 60

    # -----------------------------------------------------------------------
    # Generation / LLM
    # -----------------------------------------------------------------------

    GENERATION_BACKEND: str = "groq"
    GENERATION_MODEL: str = "openai/gpt-oss-120b"

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    OPENAI_API_KEY: str | None = None
    OPENAI_API_URL: str | None = None

    GEMINI_API_KEY: str | None = None
    GEMINI_LIVE_MODEL: str = "gemini-3.1-flash-live-preview"
    GEMINI_VOICE: str = "Aoede"

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------

    EMBEDDING_BACKEND: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_MODEL_ID: str | None = None

    # -----------------------------------------------------------------------
    # Reranking
    # -----------------------------------------------------------------------

    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str | None = None

    # -----------------------------------------------------------------------
    # Qdrant
    # -----------------------------------------------------------------------

    QDRANT_CLUSTER_ENDPOINT: str | None = None
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "properties"

    # -----------------------------------------------------------------------
    # Other retrieval / AI providers
    # -----------------------------------------------------------------------

    COHERE_API_KEY: str | None = None

    # -----------------------------------------------------------------------
    # WhatsApp
    # -----------------------------------------------------------------------

    WHATSAPP_VERIFY_TOKEN: str | None = None
    WHATSAPP_APP_SECRET: str | None = None
    WHATSAPP_ACCESS_TOKEN: str | None = None
    WHATSAPP_PHONE_NUMBER_ID: str | None = None
    WHATSAPP_API_VERSION: str = "v23.0"
    WHATSAPP_TRANSCRIPTION_MODEL: str = "whisper-large-v3-turbo"
    # Shared bearer secret for the optional open-wa gateway (channels/whatsapp.ts).
    WHATSAPP_GATEWAY_SECRET: str | None = None

    # -----------------------------------------------------------------------
    # PostgreSQL
    # -----------------------------------------------------------------------

    POSTGRES_USERNAME: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_MAIN_DATABASE: str | None = None

    # -----------------------------------------------------------------------
    # Vector database
    # -----------------------------------------------------------------------

    VECTOR_DB_BACKEND: str = "qdrant"
    VECTOR_DB_PATH: str | None = None
    VECTOR_DB_DISTANCE_METHOD: str = "cosine"

    # -----------------------------------------------------------------------
    # File / ingestion settings
    # -----------------------------------------------------------------------

    FILE_ALLOWED_TYPES: list[str] = Field(default_factory=list)
    FILE_MAX_SIZE: int | None = None
    FILE_DEFAULT_CHUNK_SIZE: int = 1500
    FILE_CHUNK_OVERLAP: int = 100


    @field_validator("FILE_MAX_SIZE", "POSTGRES_PORT", mode="before")
    @classmethod
    def blank_is_unset(cls, value: object) -> object:
        # ".env.example" ships blank optional numbers ("FILE_MAX_SIZE=").
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""
    return Settings()
