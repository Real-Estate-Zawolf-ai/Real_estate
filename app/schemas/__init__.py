"""Validated request and response models used by API routes."""

from .customer import CustomerCreate, CustomerRead
from .interaction import ChatRequest, ChatResponse
from .lead import LeadCaptureRequest, LeadQualification, LeadScoreResponse
from .property import PropertyRecord, PropertySearchRequest, PropertySearchResponse

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "CustomerCreate",
    "CustomerRead",
    "LeadQualification",
    "LeadCaptureRequest",
    "LeadScoreResponse",
    "PropertyRecord",
    "PropertySearchRequest",
    "PropertySearchResponse",
]
