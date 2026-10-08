"""HTTP/WebSocket routes for the website voice experience."""

from fastapi import APIRouter, HTTPException, Request, WebSocket

from ...api.sessions import issue_voice_token, origin_allowed
from ...channels.website import handle_website_voice
from ...helpers.config import get_settings
from ...schemas.interaction import SessionRequest
from .chat import resolve_session


router = APIRouter(tags=["voice"])


@router.post("/voice/token")
async def create_voice_token(request: SessionRequest, http_request: Request) -> dict:
    """Issue a single-use, short-lived token for opening the voice WebSocket.

    The token is bound to the same session as the text chat, so a call
    continues the typed conversation. Provider keys never reach the browser.
    """
    if not get_settings().GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Voice is not configured")
    origin = http_request.headers.get("origin")
    if origin and not origin_allowed(origin):
        raise HTTPException(status_code=403, detail="Origin not allowed")
    session_id, raw = resolve_session(request.session_id)
    return {
        "session_id": session_id,
        "token": issue_voice_token(raw, request.language),
        "ws_path": "/website/voice",
        "expires_in": get_settings().VOICE_TOKEN_TTL_SECONDS,
    }


@router.websocket("/website/voice")
async def website_voice(websocket: WebSocket) -> None:
    """Open a bidirectional website voice session."""
    await handle_website_voice(websocket)
