"""FastAPI application entry point.

Run locally with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .api.routes import (
    chat_router,
    documents_router,
    leads_router,
    properties_router,
    voice_router,
    webhooks_router,
)
from starlette.concurrency import run_in_threadpool

from .database.vector_db import create_client
from .workflows.rag import answer_query


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Real Estate Assistant API",
    version="0.1.0",
    description="API for the real-estate assistant.",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# Keep browser clients usable during early development.
# Restrict this list to the deployed frontend origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(voice_router)
app.include_router(webhooks_router)
app.include_router(chat_router)
app.include_router(properties_router)
app.include_router(documents_router)
app.include_router(leads_router)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIR),
        name="frontend-assets",
    )


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

RequirementValue = Union[
    str,
    int,
    float,
    bool,
    List[str],
    None,
]


class RAGRequest(BaseModel):
    """Customer query and extracted property requirements."""

    query: str = Field(
        min_length=1,
        max_length=2000,
    )

    requirements: Dict[str, RequirementValue] = Field(
        default_factory=dict,
    )

    # Number of final properties returned to the customer / LLM.
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # Number of candidates retrieved before reranking.
    retrieval_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )


# ---------------------------------------------------------------------------
# RAG endpoint
# ---------------------------------------------------------------------------

@app.post("/rag/search", tags=["rag"])
async def rag_search(request: RAGRequest) -> dict[str, Any]:
    """
    Run the complete RAG retrieval and generation pipeline.

    Pipeline:

        Customer Query
            ↓
        Requirements
            ↓
        Metadata Filter
            ↓
        Dense / Hybrid Retrieval
            ↓
        Top-N Candidates
            ↓
        Cross-Encoder Reranking
            ↓
        Final Top-K
            ↓
        Grounded Generation
    """

    try:
        return await run_in_threadpool(
            answer_query,
            request.query,
            request.requirements,
            top_k=request.top_k,
            retrieval_top_k=request.retrieval_top_k,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG workflow is unavailable",
        ) from error


# ---------------------------------------------------------------------------
# Frontend route
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def website() -> FileResponse:
    """Serve the public real-estate website."""

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend is not available",
        )

    return FileResponse(index_file)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Return basic API information."""

    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health/ready", tags=["system"])
async def readiness() -> dict[str, Any]:
    """Verify that the vector database required by the RAG workflow is reachable."""
    client = create_client()
    try:
        collections = await run_in_threadpool(client.get_collections)
        return {
            "status": "ready",
            "qdrant": "reachable",
            "collections": [item.name for item in collections.collections],
        }
    except Exception as error:
        raise HTTPException(status_code=503, detail="Qdrant is unavailable") from error
    finally:
        client.close()
