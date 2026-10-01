"""HTTP/WebSocket routes for the website voice experience."""

from fastapi import APIRouter, WebSocket

from app.channels.website import handle_website_voice


router = APIRouter(prefix="/website", tags=["website"])


@router.websocket("/voice")
async def website_voice(websocket: WebSocket) -> None:
    """Open a bidirectional website voice session."""
    await handle_website_voice(websocket)
