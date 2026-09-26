from typing import List
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# Chunking configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into meaningful chunks using recursive character splitting.

    The splitter tries to preserve structure by splitting on:
    paragraphs -> lines -> sentences -> words -> characters.

    Args:
        text: Input text.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of non-empty text chunks.

    Raises:
        ValueError: If chunk_size or chunk_overlap is invalid.
    """

    # Validate input
    if not text or not text.strip():
        logger.warning("Chunking skipped: empty text received")
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    logger.info(
        "Starting text chunking | text_length=%d | chunk_size=%d | chunk_overlap=%d",
        len(text),
        chunk_size,
        chunk_overlap,
    )

    try:
        # Create recursive splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                " ",     # Words
                "",      # Characters
            ],
            length_function=len,
            is_separator_regex=False,
        )

        # Split text
        chunks = splitter.split_text(text)

        # Remove empty chunks
        valid_chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

        logger.info(
            "Text chunking completed | chunks=%d | text_length=%d",
            len(valid_chunks),
            len(text),
        )

        return valid_chunks

    except Exception:
        logger.exception(
            "Text chunking failed | text_length=%d",
            len(text),
        )
        raise
