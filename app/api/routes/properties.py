"""Development property-catalogue endpoints."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.api.ratelimit import limit_search
from app.api.security import require_admin
from app.schemas.property import PropertyRecord, PropertySearchRequest, PropertySearchResponse
from app.workflows.rag import index_chunks, list_indexed_properties, search_index


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/properties", tags=["properties"])

# Every visitor loads the catalogue; one Qdrant scroll per half minute is plenty
# and keeps the free-tier cluster (and its request quota) out of the hot path.
_CATALOGUE_TTL_SECONDS = 30
_catalogue_cache: dict = {"at": 0.0, "items": None}


def reset_catalogue_cache() -> None:
    _catalogue_cache.update(at=0.0, items=None)


@router.get("", response_model=list[dict])
async def list_properties(request: Request) -> list[dict]:
    limit_search(request)
    now = time.monotonic()
    if _catalogue_cache["items"] is not None and now - _catalogue_cache["at"] < _CATALOGUE_TTL_SECONDS:
        return _catalogue_cache["items"]
    try:
        items = await run_in_threadpool(list_indexed_properties)
    except Exception as error:
        logger.exception("Property catalogue failed")
        raise HTTPException(status_code=503, detail="Property catalogue is unavailable") from error
    _catalogue_cache.update(at=now, items=items)
    return items


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
async def add_property(property_record: PropertyRecord) -> dict:
    record = property_record.model_dump(exclude_none=True)
    metadata = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k not in {"metadata", "text"}}}
    text = record.get("text") or " | ".join(f"{key}: {value}" for key, value in metadata.items())
    chunk = {"id": f"property-{record['property_id']}", "text": text, "metadata": metadata}
    try:
        await run_in_threadpool(index_chunks, [chunk])
    except Exception as error:
        logger.exception("Property indexing failed")
        raise HTTPException(status_code=503, detail="Property indexing failed") from error
    reset_catalogue_cache()  # the new listing should show up immediately
    return record


@router.post("/search", response_model=PropertySearchResponse)
async def search_properties(request: PropertySearchRequest, http_request: Request) -> PropertySearchResponse:
    limit_search(http_request)
    try:
        candidates = await run_in_threadpool(
            search_index,
            request.query or "property",
            request.filters,
            top_k=request.top_k,
            retrieval_top_k=max(request.top_k * 4, 20),
        )
    except Exception as error:
        logger.exception("Property search failed")
        raise HTTPException(status_code=503, detail="Property search is unavailable") from error
    results = [
        {
            "property_id": item.property_id,
            "text": item.text,
            "metadata": item.metadata,
            "source_file": item.source_file,
            "score": item.combined_score,
        }
        for item in candidates
    ]
    return PropertySearchResponse(results=results, count=len(results))
