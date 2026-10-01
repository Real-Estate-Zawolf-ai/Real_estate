from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Generate dense text embeddings using:

        intfloat/multilingual-e5-small

    The model produces 384-dimensional normalized embeddings suitable for
    multilingual semantic search, including Arabic and English.

    E5 models use different prefixes for queries and documents:

        query:   <user query>
        passage: <document/chunk>

    The same model and prefixes must be used consistently during indexing
    and retrieval.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        logger.info(
            "Loading embedding model | model=%s | device=%s",
            model_name,
            device or "auto",
        )

        if device:
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )
        else:
            self.model = SentenceTransformer(model_name)

        logger.info(
            "Embedding model loaded successfully | model=%s",
            model_name,
        )

    # -----------------------------------------------------------------------
    # Query embedding
    # -----------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a user query.

        E5 query format:

            query: <user query>
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        formatted_query = f"{QUERY_PREFIX}{query.strip()}"

        embedding = self.model.encode(
            formatted_query,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------------------------
    # Single document embedding
    # -----------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single document/chunk.

        E5 passage format:

            passage: <document text>
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        formatted_text = f"{PASSAGE_PREFIX}{text.strip()}"

        embedding = self.model.encode(
            formatted_text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------------------------
    # Batch document embeddings
    # -----------------------------------------------------------------------

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents/chunks.

        Empty or whitespace-only texts are ignored.

        The returned embeddings preserve the order of the valid input texts.
        """
        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not cleaned_texts:
            return []

        formatted_texts = [
            f"{PASSAGE_PREFIX}{text}"
            for text in cleaned_texts
        ]

        embeddings = self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    # -----------------------------------------------------------------------
    # Model information
    # -----------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        return EMBEDDING_DIMENSION
