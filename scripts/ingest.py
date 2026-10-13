"""Ingest files, embed their chunks, and index them in Qdrant.

Run this on a developer machine (it needs ``requirements-ingest.txt`` for the
PDF / spreadsheet / OCR parsers), pointing at Qdrant Cloud through the same
environment variables the API uses::

    pip install -r requirements-ingest.txt
    export QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333 QDRANT_API_KEY=... GEMINI_API_KEY=...
    python scripts/ingest.py data/raw --recursive --rebuild

Queries and documents must be embedded with the same backend, model and
dimension, so use the same ``EMBEDDING_*`` settings here as on the server.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.vector_db import (  # noqa: E402
    create_client,
    ensure_collection,
    rebuild_collection,
    upsert_chunks,
)
from app.helpers.config import get_settings  # noqa: E402
from app.rag.embeddings import EmbeddingModel  # noqa: E402

try:
    from app.rag.ingestion.pipeline import ingest_directory, ingest_file
except ImportError as error:  # the lean API install has no document parsers
    raise SystemExit(
        f"Document parsers are not installed ({error.name}). Run: pip install -r requirements-ingest.txt"
    ) from error

logger = logging.getLogger("ingest")


def _chunks_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect valid chunk records from a file or directory result."""
    if "results" in result:
        return [
            chunk
            for file_result in result["results"]
            if file_result.get("status") == "success"
            for chunk in file_result.get("chunks", [])
        ]
    return list(result.get("chunks", []))


def index_path(
    path: str,
    *,
    collection_name: str | None = None,
    recursive: bool = False,
    rebuild: bool = False,
) -> int:
    """Run ingestion → shared embeddings → Qdrant upsert for a path."""
    source = Path(path)
    if source.is_dir():
        result = ingest_directory(path, recursive=recursive)
    else:
        result = ingest_file(path)

    chunks = _chunks_from_result(result)
    valid_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]
    if not valid_chunks:
        logger.info("No text chunks to index for %s", path)
        return 0

    # Embed only the records represented in this ordered list; embed_texts
    # preserves their order.
    embedder = EmbeddingModel()
    settings = get_settings()
    logger.info("Embedding %d chunks with %s:%s (%d dims)", len(valid_chunks), embedder.backend_name, embedder.model_name, embedder.dimension)
    vectors = embedder.embed_texts([chunk["text"] for chunk in valid_chunks])
    if len(vectors) != len(valid_chunks):
        raise RuntimeError("Embedding output count does not match chunk count")

    name = collection_name or settings.QDRANT_COLLECTION
    client = create_client()
    try:
        if rebuild:
            rebuild_collection(client, name)
        else:
            ensure_collection(client, name)
        return upsert_chunks(client, name, valid_chunks, vectors)
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A supported input file or directory")
    parser.add_argument("--collection", help="Qdrant collection name")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    count = index_path(
        args.path,
        collection_name=args.collection,
        recursive=args.recursive,
        rebuild=args.rebuild,
    )
    print(f"Indexed {count} chunks into {args.collection or get_settings().QDRANT_COLLECTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
