"""Alembic migration environment for the real-estate assistant."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.helpers.config import get_settings
from app.database.models import Base


# ============================================================
# Alembic Configuration
# ============================================================

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ============================================================
# SQLAlchemy Metadata
# ============================================================

target_metadata = Base.metadata


# ============================================================
# Database URL
# ============================================================

def _build_database_url() -> str:
    """Build the PostgreSQL URL from application settings."""

    settings = get_settings()

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
            "Missing PostgreSQL configuration: "
            + ", ".join(missing)
        )

    return (
        f"postgresql+asyncpg://"
        f"{username}:{password}@{host}:{port}/{database}"
    )


# ============================================================
# Offline Migrations
# ============================================================

def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""

    url = _build_database_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================
# Migration Runner
# ============================================================

def do_run_migrations(connection: Connection) -> None:
    """Configure Alembic and run migrations."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================
# Async Migrations
# ============================================================

async def run_async_migrations() -> None:
    """Create an async engine and run migrations."""

    database_url = _build_database_url()

    connectable = async_engine_from_config(
        {
            "sqlalchemy.url": database_url,
        },
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


# ============================================================
# Online Migrations
# ============================================================

def run_migrations_online() -> None:
    """Run migrations using the async PostgreSQL engine."""

    asyncio.run(run_async_migrations())


# ============================================================
# Entry Point
# ============================================================

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()