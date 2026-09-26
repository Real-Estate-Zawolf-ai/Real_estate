from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_metadata(
    file_path: str,
    *,
    source_type: str | None = None,
    property_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build standardized metadata for an ingested document.

    Metadata is attached to documents/chunks and can later be used for:
    - filtering
    - retrieval
    - tracing
    - source attribution
    - property-level grouping
    """

    path = Path(file_path)

    metadata: dict[str, Any] = {
        "source": path.name,
        "source_path": str(path),
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
    }

    if source_type:
        metadata["source_type"] = source_type

    if property_id:
        metadata["property_id"] = property_id

    if extra_metadata:
        metadata.update(extra_metadata)

    logger.debug(
        "Metadata created | file=%s | property_id=%s | keys=%s",
        file_path,
        property_id,
        list(metadata.keys()),
    )

    return metadata


def add_chunk_metadata(
    metadata: dict[str, Any],
    *,
    chunk_index: int,
    total_chunks: int | None = None,
) -> dict[str, Any]:
    """
    Add chunk-level metadata to an existing metadata dictionary.
    """

    chunk_metadata = metadata.copy()

    chunk_metadata["chunk_index"] = chunk_index

    if total_chunks is not None:
        chunk_metadata["total_chunks"] = total_chunks

    return chunk_metadata