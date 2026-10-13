"""Synchronous SQLAlchemy engine and sessions shared by every service.

The agent graph, CRM, bookings and webhook guards are synchronous and run in
worker threads, so they share one pooled sync engine. PostgreSQL (Neon) is the
production database; outside production an unset ``DATABASE_URL`` falls back to
a local SQLite file so tests and local development need no server.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from threading import Lock
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import Base
from app.database.url import is_sqlite, redacted, sync_url
from app.helpers.config import get_settings


logger = logging.getLogger(__name__)

_engine: Engine | None = None
_factory: sessionmaker[Session] | None = None
_guard = Lock()


def _build_engine() -> Engine:
    settings = get_settings()
    url = sync_url(settings)

    if is_sqlite(url):
        in_memory = not url.database or url.database == ":memory:"
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if in_memory else None,
            future=True,
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if not in_memory:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        logger.info("Database: SQLite (%s) - development/test only", url.database or ":memory:")
        return engine

    engine = create_engine(
        url,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        # Neon suspends idle computes after ~5 minutes and drops their sockets.
        pool_recycle=240,
        pool_timeout=15,
        connect_args={
            "connect_timeout": 10,
            # No server-side prepared statements: required behind a pooler
            # (pgbouncer transaction mode) and harmless otherwise.
            "prepare_threshold": None,
        },
        future=True,
    )
    logger.info("Database: PostgreSQL %s", redacted(url))
    return engine


def get_engine() -> Engine:
    global _engine, _factory
    if _engine is None:
        with _guard:
            if _engine is None:
                _engine = _build_engine()
                _factory = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False, future=True)
    return _engine


def reset_engine() -> None:
    """Dispose the engine so the next call rebuilds it from current settings."""
    global _engine, _factory
    with _guard:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _factory = None


def new_session() -> Session:
    get_engine()
    assert _factory is not None
    return _factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transaction: commits on success, rolls back on any error."""
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_postgres() -> bool:
    return not is_sqlite(sync_url())


def create_all() -> None:
    """Create every table from the ORM models (SQLite dev/tests only).

    PostgreSQL schemas are managed exclusively by Alembic migrations.
    """
    engine = get_engine()
    if not is_sqlite(engine.url):
        raise RuntimeError("create_all() is for SQLite only; run `alembic upgrade head` for PostgreSQL")
    Base.metadata.create_all(engine)


def drop_all() -> None:
    engine = get_engine()
    if not is_sqlite(engine.url):
        raise RuntimeError("drop_all() is for SQLite only")
    Base.metadata.drop_all(engine)


def check_database() -> tuple[bool, str | None]:
    """Cheap connectivity probe for /health/ready (no secrets in the message)."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, None
    except Exception as error:
        return False, type(error).__name__
