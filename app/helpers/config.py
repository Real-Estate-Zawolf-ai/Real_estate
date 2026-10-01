"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "realEstate"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""
    return Settings()
