"""PostgreSQL database connection and session management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.helpers.config import get_settings


# ============================================================
# Configuration
# ============================================================

settings = get_settings()


def _build_database_url() -> str:
    """Build the async PostgreSQL SQLAlchemy connection URL."""
    username = settings.POSTGRES_USERNAME
    password = settings.POSTGRES_PASSWORD
    host = settings.POSTGRES_HOST
    port = settings.POSTGRES_PORT
    database = settings.POSTGRES_MAIN_DATABASE

    missing = []

    if not username:
        missing.append("POSTGRES_USERNAME")

    if not password:
        missing.append("POSTGRES_PASSWORD")

    if not host:
        missing.append("POSTGRES_HOST")

    if not port:
        missing.append("POSTGRES_PORT")

    if not database:
        missing.append("POSTGRES_MAIN_DATABASE")

    if missing:
        raise RuntimeError(
            "Missing PostgreSQL configuration: " + ", ".join(missing)
        )

    return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"


# ============================================================
# Database URL
# ============================================================

DATABASE_URL = _build_database_url()


# ============================================================
# Engine
# ============================================================

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=False,
)


# ============================================================
# Session Factory
# ============================================================

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# FastAPI Dependency
# ============================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an async database session for FastAPI dependencies.

    The session is rolled back if an exception occurs and is always
    closed after the request finishes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ============================================================
# Transaction Context Manager
# ============================================================

@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an async transactional database session.

    Successful operations are committed automatically.
    Failed operations are rolled back automatically.

    Example:
        async with db_session() as db:
            db.add(model)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ============================================================
# Health Check
# ============================================================

async def check_database_connection() -> bool:
    """
    Check whether PostgreSQL is reachable.

    Returns:
        True when the database connection succeeds.
        False when the connection fails.
    """
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        return True

    except Exception:
        return False


# ============================================================
# Shutdown
# ============================================================

async def dispose_engine() -> None:
    """Dispose the SQLAlchemy async engine and its connection pool."""
    await engine.dispose()