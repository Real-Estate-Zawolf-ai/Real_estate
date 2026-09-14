"""Storage abstraction used by the CRM services.

This in-memory implementation is intended for development and tests. Replace
it with a PostgreSQL or external CRM adapter while retaining the service APIs.
"""

from threading import Lock
from typing import Any
from uuid import UUID


class CRMClient:
    """Thread-safe in-memory store for CRM entities."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[UUID, Any]] = {
            "customers": {},
            "leads": {},
            "interactions": {},
        }
        self._lock = Lock()

    def save(self, collection: str, entity_id: UUID, entity: Any) -> Any:
        """Create or replace an entity in a named collection."""
        with self._lock:
            self._collections[collection][entity_id] = entity
            return entity

    def get(self, collection: str, entity_id: UUID) -> Any | None:
        """Fetch one entity by identifier."""
        with self._lock:
            return self._collections[collection].get(entity_id)

    def list(self, collection: str) -> list[Any]:
        """Return a snapshot of a collection's entities."""
        with self._lock:
            return list(self._collections[collection].values())
