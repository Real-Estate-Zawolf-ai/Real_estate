"""FastAPI application entry point.

Run locally with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .api.routes import voice_router, webhooks_router
from .database.vector_db import DEFAULT_COLLECTION, create_client
from .rag.generation import generate_response
from .rag.retrieval.hybrid_search import hybrid_search
from .rag.retrieval.metadata_filter import build_metadata_filter
from .rag.retrieval.reranker import rerank


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

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be blank",
        )

    # -----------------------------------------------------------------------
    # Metadata filter
    # -----------------------------------------------------------------------

    try:
        requirements_filter = build_metadata_filter(
            request.requirements,
        )
    except (TypeError, ValueError) as error:
        logger.warning(
            "Invalid property requirements: %s",
            error,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid property requirements",
        ) from error

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    qdrant_client = None

    try:
        collection_name = os.getenv(
            "QDRANT_COLLECTION",
            DEFAULT_COLLECTION,
        )

        qdrant_client = create_client()

        candidates = hybrid_search(
            query=query,
            requirements_filter=requirements_filter,
            collection_name=collection_name,
            qdrant_client=qdrant_client,
            top_k=request.retrieval_top_k,
        )

        logger.info(
            "Property retrieval completed | query=%r | candidates=%d",
            query,
            len(candidates),
        )

    except Exception as error:
        logger.exception(
            "Property search failed | query=%r",
            query,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Property search is temporarily unavailable",
        ) from error

    finally:
        if qdrant_client is not None:
            qdrant_client.close()

    # -----------------------------------------------------------------------
    # Reranking
    # -----------------------------------------------------------------------

    try:
        reranked_candidates = rerank(
            query=query,
            candidates=candidates,
            top_k=request.top_k,
        )

        logger.info(
            "Property reranking completed | candidates=%d | final=%d",
            len(candidates),
            len(reranked_candidates),
        )

    except Exception as error:
        logger.exception(
            "Property reranking failed | query=%r",
            query,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Property reranking is temporarily unavailable",
        ) from error

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    try:
        result = generate_response(
            customer_query=query,
            candidates=reranked_candidates,
        )

        logger.info(
            "Response generation completed | cited=%d | grounded=%d",
            len(result.cited_property_ids),
            len(result.grounding_property_ids),
        )

    except Exception as error:
        logger.exception(
            "Response generation failed | query=%r",
            query,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate a response",
        ) from error

    # -----------------------------------------------------------------------
    # Response
    # -----------------------------------------------------------------------

    return {
        "answer": result.answer_text,
        "cited_property_ids": result.cited_property_ids,
        "grounding_property_ids": result.grounding_property_ids,
        "properties": [
            {
                "property_id": candidate.property_id,
                "text": candidate.text,
                "metadata": candidate.metadata,
                "source_file": candidate.source_file,
                "score": candidate.combined_score,
            }
            for candidate in reranked_candidates
        ],
    }


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