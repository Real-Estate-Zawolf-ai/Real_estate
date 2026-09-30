"""Qdrant client and collection operations for RAG indexing."""

from __future__ import annotations

import os
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.rag.embeddings import EMBEDDING_DIMENSION


DEFAULT_COLLECTION = "properties"

_POINT_ID_NAMESPACE = NAMESPACE_URL

# Keep requests comfortably below Qdrant Cloud's 32 MB request limit.
UPSERT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Payload indexes
# ---------------------------------------------------------------------------

# Fields used by metadata filtering.
#
# Qdrant requires a payload index for filtered search. Keep the schema
# explicit so the collection has predictable filter behavior.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "location": models.PayloadSchemaType.KEYWORD,
    "city": models.PayloadSchemaType.KEYWORD,
    "district": models.PayloadSchemaType.KEYWORD,
    "property_type": models.PayloadSchemaType.KEYWORD,
    "bedrooms": models.PayloadSchemaType.INTEGER,
    "bathrooms": models.PayloadSchemaType.INTEGER,
    "price": models.PayloadSchemaType.FLOAT,
    "area": models.PayloadSchemaType.FLOAT,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def create_client(
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Create a Qdrant client using the project's environment settings."""
    endpoint = (
        url
        or os.getenv("QDRANT_CLUSTER_ENDPOINT")
        or os.getenv("QDRANT_URL")
        or "http://localhost:6333"
    )

    return QdrantClient(
        url=endpoint,
        api_key=api_key if api_key is not None else os.getenv("QDRANT_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Ensure all metadata-filter fields have Qdrant payload indexes."""

    for field_name, schema_type in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
            wait=True,
        )


def ensure_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """Ensure the collection and its metadata indexes exist."""

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )

    ensure_payload_indexes(client, collection_name)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: Iterable[dict[str, Any]],
    vectors: Iterable[list[float]],
) -> int:
    """Upsert embedded chunks in batches with searchable metadata payloads."""

    chunk_list = list(chunks)
    vector_list = list(vectors)

    if len(chunk_list) != len(vector_list):
        raise ValueError(
            f"Chunk/vector count mismatch: {len(chunk_list)} chunks, "
            f"{len(vector_list)} vectors"
        )

    points: list[models.PointStruct] = []

    for chunk, vector in zip(chunk_list, vector_list):
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")

        if not chunk_id:
            raise ValueError("Each chunk must include a deterministic id")

        metadata = chunk.get("metadata") or {}

        if not isinstance(metadata, dict):
            raise TypeError("Chunk metadata must be a dictionary")

        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional vector, "
                f"got {len(vector)}"
            )

        payload = {
            **metadata,
            "text": chunk.get("text", ""),
            "id": chunk_id,
            "chunk_id": chunk_id,
            "metadata": metadata,
        }

        # Qdrant accepts unsigned integers or UUIDs as point IDs.
        # Derive a stable UUID from the original chunk ID so arbitrary
        # deterministic IDs remain repeatable across indexing runs.
        point_id = str(uuid5(_POINT_ID_NAMESPACE, chunk_id))

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    if not points:
        return 0

    total = 0

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[start : start + UPSERT_BATCH_SIZE]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

        total += len(batch)

    return total


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


def delete_collection(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Delete a collection if it exists."""

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)


def rebuild_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """Drop and recreate a collection with the configured schema."""

    delete_collection(client, collection_name)
    ensure_collection(client, collection_name)