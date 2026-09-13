"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Real Estate Assistant API",
    version="0.1.0",
    description="API for the real-estate assistant.",
)

# Keep browser clients usable during early development. Restrict this list to
# the deployed frontend origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return a lightweight liveness check."""
    return {"status": "ok"}
