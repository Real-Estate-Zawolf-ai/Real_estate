"""FastAPI route modules."""

from .chat import router as chat_router
from .documents import router as documents_router
from .leads import router as leads_router
from .properties import router as properties_router
from .voice import router as voice_router
from .webhooks import router as webhooks_router

__all__ = [
    "chat_router",
    "documents_router",
    "leads_router",
    "properties_router",
    "voice_router",
    "webhooks_router",
]
