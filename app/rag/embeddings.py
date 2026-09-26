from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384


class EmbeddingModel:
    """
    Generate dense text embeddings using BAAI/bge-m3.

    BGE-M3 produces 1024-dimensional dense embeddings
    suitable for multilingual semantic search.
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

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """
        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        return EMBEDDING_DIMENSION