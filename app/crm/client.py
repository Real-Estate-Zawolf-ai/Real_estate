"""Database handle shared by the CRM services.

The CRM used to keep customers, leads and interactions in process memory, which
lost everything on restart and split data between workers. They now live in
PostgreSQL; ``CRMClient`` only supplies transactional sessions so services stay
easy to test with a different session factory.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable

from sqlalchemy.orm import Session

from app.database.session import session_scope


class CRMClient:
    """Provides one transaction per ``with client.session()`` block."""

    def __init__(self, scope: Callable[[], AbstractContextManager[Session]] | None = None) -> None:
        self._scope = scope or session_scope

    def session(self) -> AbstractContextManager[Session]:
        return self._scope()
