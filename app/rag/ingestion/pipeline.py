"""
RAG Ingestion Pipeline.

Current responsibility:

    file
        -> file type detection
        -> loader
        -> extracted content
        -> metadata
        -> chunking
        -> chunk records

Images are only loaded here.

OCR, CV, normalization, embeddings, and vector storage
are handled by separate components and will be integrated later.

The pipeline keeps each ingestion component loosely coupled
so they can be replaced independently.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from app.rag.ingestion import (
    load_image,
    parse_office,
    parse_pdf,
    parse_spreadsheet,
    parse_text,
)
from app.rag.ingestion.chunker import chunk_text
from app.rag.ingestion.metadata import (
    add_chunk_metadata,
    build_metadata,
)

logger = logging.getLogger(__name__)


# ============================================================
# Types
# ============================================================

Loader = Callable[[str], Any]


# ============================================================
# Supported file types
# ============================================================

LOADER_REGISTRY: dict[str, Loader] = {
    # Documents
    ".pdf": parse_pdf,
    ".docx": parse_office,
    ".pptx": parse_office,

    # Text
    ".txt": parse_text,
    ".md": parse_text,
    ".markdown": parse_text,

    # Spreadsheets
    ".csv": parse_spreadsheet,
    ".xlsx": parse_spreadsheet,
    ".xls": parse_spreadsheet,

    # Images
    ".jpg": load_image,
    ".jpeg": load_image,
    ".png": load_image,
    ".webp": load_image,
    ".bmp": load_image,
    ".tiff": load_image,
    ".tif": load_image,
}


FILE_TYPE_MAP: dict[str, str] = {
    # Documents
    ".pdf": "pdf",
    ".docx": "office",
    ".pptx": "office",

    # Text
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",

    # Spreadsheets
    ".csv": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",

    # Images
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
}


# ============================================================
# File helpers
# ============================================================

def detect_file_type(file_path: str) -> str:
    """
    Detect the logical file type from its extension.

    Returns:
        pdf, office, text, spreadsheet, image, or unknown.
    """
    extension = Path(file_path).suffix.lower()
    return FILE_TYPE_MAP.get(extension, "unknown")


def is_supported(file_path: str) -> bool:
    """
    Check whether the file extension is supported.
    """
    extension = Path(file_path).suffix.lower()
    return extension in LOADER_REGISTRY


def _build_chunk_id(
    file_path: Path,
    chunk_index: int,
    property_id: str | None = None,
) -> str:
    """
    Build a deterministic identifier for a chunk.

    The same file + property + chunk index will generate
    the same identifier.
    """
    identity = (
        f"{property_id or 'no-property'}"
        f"::{file_path.resolve()}"
        f"::{chunk_index}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


# ============================================================
# Single file ingestion
# ============================================================

def ingest_file(
    file_path: str,
    *,
    property_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
) -> dict[str, Any]:
    """
    Ingest a single file.

    Current flow:

        file
            -> detect type
            -> loader
            -> metadata
            -> chunking
            -> chunk records

    Embeddings and vector storage are intentionally NOT handled
    here yet. They will be integrated after those components
    are implemented.

    Args:
        file_path:
            Path to the input file.

        property_id:
            Optional real-estate property identifier.

        extra_metadata:
            Additional metadata such as developer,
            compound, language, source, etc.

        chunk_size:
            Maximum chunk size.

        chunk_overlap:
            Number of overlapping characters.

    Returns:
        Dictionary containing:
            - status
            - file information
            - metadata
            - chunks
            - chunk count
    """

    path = Path(file_path)

    logger.info(
        "Starting file ingestion | file=%s",
        file_path,
    )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not path.exists():
        logger.error(
            "File does not exist | file=%s",
            file_path,
        )
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        logger.error(
            "Path is not a file | path=%s",
            file_path,
        )
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    # --------------------------------------------------------
    # Detect file type
    # --------------------------------------------------------

    extension = path.suffix.lower()

    if extension not in LOADER_REGISTRY:
        logger.warning(
            "Unsupported file type | file=%s | extension=%s",
            file_path,
            extension,
        )

        return {
            "status": "unsupported",
            "file": str(path),
            "file_type": "unknown",
            "reason": f"Unsupported extension: {extension}",
            "chunks": [],
            "chunks_count": 0,
        }

    source_type = detect_file_type(str(path))

    logger.info(
        "Detected file type | file=%s | source_type=%s",
        file_path,
        source_type,
    )

    # --------------------------------------------------------
    # Load file
    # --------------------------------------------------------

    loader = LOADER_REGISTRY[extension]

    try:
        raw_content = loader(str(path))

    except Exception:
        logger.exception(
            "File loading failed | file=%s",
            file_path,
        )
        raise

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    document_metadata = build_metadata(
        str(path),
        source_type=source_type,
        property_id=property_id,
        extra_metadata=extra_metadata,
    )

    # ========================================================
    # IMAGE FLOW
    # ========================================================

    if source_type == "image":

        logger.info(
            "Image loaded successfully | file=%s",
            file_path,
        )

        return {
            "status": "image_loaded",
            "file": str(path),
            "file_type": source_type,
            "property_id": property_id,
            "chunks": [],
            "chunks_count": 0,
            "metadata": document_metadata,
            "content": raw_content,
            "message": (
                "Image loaded successfully. "
                "OCR/CV processing is required before "
                "text chunking."
            ),
        }

    # ========================================================
    # TEXT FLOW
    # ========================================================

    if not isinstance(raw_content, str):
        logger.error(
            "Loader returned unsupported content type | "
            "file=%s | type=%s",
            file_path,
            type(raw_content).__name__,
        )

        raise TypeError(
            f"Expected loader to return str, "
            f"got {type(raw_content).__name__}"
        )

    text = raw_content.strip()

    # --------------------------------------------------------
    # Empty content
    # --------------------------------------------------------

    if not text:

        logger.warning(
            "No text extracted from file | file=%s",
            file_path,
        )

        return {
            "status": "empty",
            "file": str(path),
            "file_type": source_type,
            "property_id": property_id,
            "chunks": [],
            "chunks_count": 0,
            "metadata": document_metadata,
        }

    logger.info(
        "Text extracted | file=%s | characters=%d",
        file_path,
        len(text),
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:

        logger.warning(
            "No chunks generated | file=%s",
            file_path,
        )

        return {
            "status": "empty",
            "file": str(path),
            "file_type": source_type,
            "property_id": property_id,
            "chunks": [],
            "chunks_count": 0,
            "metadata": document_metadata,
        }

    logger.info(
        "Chunks generated | file=%s | chunks=%d",
        file_path,
        len(chunks),
    )

    # --------------------------------------------------------
    # Build chunk records
    # --------------------------------------------------------

    total_chunks = len(chunks)

    chunk_records: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):

        chunk_metadata = add_chunk_metadata(
            document_metadata,
            chunk_index=index,
            total_chunks=total_chunks,
        )

        chunk_id = _build_chunk_id(
            path,
            index,
            property_id,
        )

        chunk_records.append(
            {
                "id": chunk_id,
                "text": chunk,
                "metadata": chunk_metadata,
            }
        )

    # --------------------------------------------------------
    # Return ingestion result
    # --------------------------------------------------------

    logger.info(
        "File ingestion completed | "
        "file=%s | chunks=%d",
        file_path,
        total_chunks,
    )

    return {
        "status": "success",
        "file": str(path),
        "file_type": source_type,
        "property_id": property_id,
        "chunks": chunk_records,
        "chunks_count": total_chunks,
        "metadata": document_metadata,
    }


# ============================================================
# Directory ingestion
# ============================================================

def ingest_directory(
    directory_path: str,
    *,
    property_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    recursive: bool = False,
) -> dict[str, Any]:
    """
    Ingest all supported files inside a directory.

    Current flow:

        directory
            -> files
            -> ingest_file()
            -> chunks

    Embeddings and vector storage are not performed here yet.
    """

    directory = Path(directory_path)

    logger.info(
        "Starting directory ingestion | directory=%s",
        directory_path,
    )

    # --------------------------------------------------------
    # Validate directory
    # --------------------------------------------------------

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory_path}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Path is not a directory: {directory_path}"
        )

    # --------------------------------------------------------
    # Collect supported files
    # --------------------------------------------------------

    if recursive:

        files = [
            path
            for path in directory.rglob("*")
            if path.is_file()
            and is_supported(str(path))
        ]

    else:

        files = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and is_supported(str(path))
        ]

    logger.info(
        "Supported files discovered | "
        "directory=%s | files=%d",
        directory_path,
        len(files),
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = len(files)

    successful = 0
    skipped = 0
    failed = 0
    total_chunks = 0

    results: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for path in files:

        logger.info(
            "Processing file | file=%s",
            path,
        )

        try:

            result = ingest_file(
                str(path),
                property_id=property_id,
                extra_metadata=extra_metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            results.append(result)

            status = result.get("status")

            if status == "success":

                successful += 1

                total_chunks += result.get(
                    "chunks_count",
                    0,
                )

            elif status in {
                "unsupported",
                "empty",
                "image_loaded",
            }:

                skipped += 1

        except Exception as exc:

            failed += 1

            logger.exception(
                "File ingestion failed | file=%s",
                path,
            )

            results.append(
                {
                    "status": "failed",
                    "file": str(path),
                    "error": str(exc),
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "status": "completed",
        "directory": str(directory),
        "total_files": total,
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "total_chunks": total_chunks,
        "results": results,
    }

    logger.info(
        "Directory ingestion completed | "
        "directory=%s | total=%d | "
        "successful=%d | skipped=%d | "
        "failed=%d | chunks=%d",
        directory_path,
        total,
        successful,
        skipped,
        failed,
        total_chunks,
    )

    return summary