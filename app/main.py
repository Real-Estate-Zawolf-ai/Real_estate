"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import voice_router, webhooks_router


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

app.include_router(voice_router)
app.include_router(webhooks_router)

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")


@app.get("/", include_in_schema=False)
async def website() -> FileResponse:
    """Serve the public real-estate website."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", tags=["system"])
async def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
    }
