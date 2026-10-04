"""Development property-catalogue endpoints."""

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.schemas.property import PropertyRecord, PropertySearchRequest, PropertySearchResponse
from app.workflows.rag import index_chunks, list_indexed_properties, search_index


router = APIRouter(prefix="/properties", tags=["properties"])


@router.get("", response_model=list[dict])
async def list_properties() -> list[dict]:
    try:
        return await run_in_threadpool(list_indexed_properties)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Property catalogue is unavailable") from error


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def add_property(property_record: PropertyRecord) -> dict:
    record = property_record.model_dump(exclude_none=True)
    metadata = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k not in {"metadata", "text"}}}
    text = record.get("text") or " | ".join(f"{key}: {value}" for key, value in metadata.items())
    chunk = {"id": f"property-{record['property_id']}", "text": text, "metadata": metadata}
    try:
        await run_in_threadpool(index_chunks, [chunk])
    except Exception as error:
        raise HTTPException(status_code=503, detail="Property indexing failed") from error
    return record


@router.post("/search", response_model=PropertySearchResponse)
async def search_properties(request: PropertySearchRequest) -> PropertySearchResponse:
    try:
        candidates = await run_in_threadpool(
            search_index,
            request.query or "property",
            request.filters,
            top_k=request.top_k,
            retrieval_top_k=max(request.top_k * 4, 20),
        )
    except Exception as error:
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
