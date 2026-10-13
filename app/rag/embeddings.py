"""Text embeddings for property search.

Two interchangeable backends, selected with ``EMBEDDING_BACKEND``:

``gemini`` (default)
    Google's hosted ``gemini-embedding-001`` - multilingual (Arabic + English),
    no local model, so it fits a 512 MB free instance.
``local``
    ``sentence-transformers`` (``intfloat/multilingual-e5-small`` by default).
    Needs ``requirements-ingest.txt`` and roughly 1 GB of RAM.

Documents and queries must be embedded with the *same* backend, model and
dimension; the Qdrant collection is created with ``EMBEDDING_DIMENSION`` and the
app refuses to search a collection of a different size (see ``vector_db``).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Sequence

from app.helpers.config import get_settings


logger = logging.getLogger(__name__)

# E5 models expect these prefixes; other models do not.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_GEMINI_BATCH = 100  # embed_content accepts up to 100 texts per request
_RETRY_STATUSES = {429, 500, 502, 503, 504}


def embedding_dimension() -> int:
    """Vector size of the configured embedding setup (Qdrant collection size)."""
    return get_settings().EMBEDDING_DIMENSION


# Kept for modules that imported the constant before backends existed.
EMBEDDING_DIMENSION = embedding_dimension()


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [float(value) / norm for value in vector] if norm else [float(value) for value in vector]


class EmbeddingError(RuntimeError):
    """The embedding provider failed or returned something unusable."""


# ---------------------------------------------------------------------------
# Gemini (hosted)
# ---------------------------------------------------------------------------


class _GeminiBackend:

    def __init__(self, model_name: str, dimension: int) -> None:
        settings = get_settings()
        if not settings.GEMINI_API_KEY:
            raise EmbeddingError("GEMINI_API_KEY is not configured (EMBEDDING_BACKEND=gemini)")
        from google import genai

        self.model_name = model_name
        self.dimension = dimension
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _GEMINI_BATCH):
            batch = texts[start : start + _GEMINI_BATCH]
            response = self._call_with_retry(
                lambda batch=batch: self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimension),
                )
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise EmbeddingError(f"Gemini returned {len(embeddings)} embeddings for {len(batch)} texts")
            for item in embeddings:
                values = list(item.values or [])
                if len(values) != self.dimension:
                    raise EmbeddingError(f"Gemini returned a {len(values)}-dimensional vector, expected {self.dimension}")
                vectors.append(_normalize(values))
        return vectors

    @staticmethod
    def _call_with_retry(call, attempts: int = 4):
        """Free-tier quotas answer 429 under bursts; back off and retry."""
        delay = 1.0
        for attempt in range(1, attempts + 1):
            try:
                return call()
            except Exception as error:
                status = getattr(error, "code", None) or getattr(error, "status_code", None)
                if attempt == attempts or status not in _RETRY_STATUSES:
                    raise EmbeddingError(f"Gemini embedding request failed ({status or type(error).__name__})") from error
                logger.warning("Gemini embeddings %s; retrying in %.0fs", status, delay)
                time.sleep(delay)
                delay *= 2

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_QUERY")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")


# ---------------------------------------------------------------------------
# Local sentence-transformers
# ---------------------------------------------------------------------------


class _LocalBackend:

    def __init__(self, model_name: str, device: str | None, dimension: int) -> None:
        # Keep module imports lightweight for API startup and unit tests: the ML
        # stack is only needed when an embedding is actually requested.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError(
                "EMBEDDING_BACKEND=local needs sentence-transformers (pip install -r requirements-ingest.txt)"
            ) from error

        logger.info("Loading embedding model | model=%s | device=%s", model_name, device or "auto")
        self.model_name = model_name
        self.dimension = dimension
        self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        actual = self.model.get_sentence_embedding_dimension()
        if actual and actual != dimension:
            raise EmbeddingError(
                f"{model_name} produces {actual}-dimensional vectors but EMBEDDING_DIMENSION={dimension}; set them equal"
            )

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([QUERY_PREFIX + text for text in texts], normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            [PASSAGE_PREFIX + text for text in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Embeds queries and documents with the configured backend."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.dimension_value = settings.EMBEDDING_DIMENSION
        self.backend_name = settings.EMBEDDING_BACKEND
        if self.backend_name == "local":
            self.model_name = model_name or settings.EMBEDDING_LOCAL_MODEL
            self._backend = _LocalBackend(self.model_name, device, self.dimension_value)
        else:
            self.model_name = model_name or settings.EMBEDDING_MODEL
            self._backend = _GeminiBackend(self.model_name, self.dimension_value)

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query for retrieval."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        return self._backend.embed_queries([query.strip()])[0]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single document/chunk."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        return self._backend.embed_documents([text.strip()])[0]

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        """Embed documents; blank texts are skipped and order is preserved."""
        cleaned = [text.strip() for text in texts or [] if isinstance(text, str) and text.strip()]
        return self._backend.embed_documents(cleaned) if cleaned else []

    @property
    def dimension(self) -> int:
        return self.dimension_value
