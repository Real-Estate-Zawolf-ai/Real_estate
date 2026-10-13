"""Document ingestion endpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.security import require_admin
from app.helpers.config import get_settings
from app.workflows.rag import chunks_from_ingestion, index_chunks


router = APIRouter(prefix="/documents", tags=["documents"])


def _load_ingestion():
    """Import the parsing stack on demand.

    PDF/OCR/spreadsheet parsers live in ``requirements-ingest.txt`` and are not
    installed on the lean production API, which must still boot without them.
    """
    try:
        from app.rag.ingestion.pipeline import ingest_file, is_supported
    except ImportError as error:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Document ingestion is not installed on this deployment; run scripts/ingest.py offline",
        ) from error
    return ingest_file, is_supported


@router.post("/ingest", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
async def ingest_document(
    file: UploadFile = File(...),
    image_kind: str | None = Form(default=None),
) -> dict[str, Any]:
    ingest_file, is_supported = _load_ingestion()
    filename = Path(file.filename or "upload").name
    if not is_supported(filename):
        raise HTTPException(status_code=415, detail="Unsupported document type")
    if image_kind is not None and image_kind not in {"photo", "floorplan"}:
        raise HTTPException(status_code=400, detail="image_kind must be 'photo' or 'floorplan'")

    content = await file.read()
    max_size = get_settings().FILE_MAX_SIZE
    if max_size is not None and len(content) > max_size:
        raise HTTPException(status_code=413, detail="Document exceeds FILE_MAX_SIZE")

    suffix = Path(filename).suffix.lower()
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(content)
            temporary_path = temporary.name
        result = await run_in_threadpool(
            ingest_file,
            temporary_path,
            extra_metadata={
                "original_file_name": filename,
                **({"image_kind": image_kind} if image_kind else {}),
            },
        )
        try:
            indexed_count = await run_in_threadpool(
                index_chunks,
                chunks_from_ingestion(result),
            )
        except Exception as error:
            raise HTTPException(status_code=503, detail="Document was parsed but indexing failed") from error
        result["file"] = filename
        result["indexed_count"] = indexed_count
        return result
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
