"""Alembic migration environment for the real-estate assistant.

Connection settings come from the application settings (``DATABASE_URL`` or the
``POSTGRES_*`` parts), the same way the API reads them, so there is a single
place to configure the database. Migrations run over the synchronous psycopg
driver; for Neon use the *direct* (non-pooled) connection string.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.database.models import Base
from app.database.url import sync_url


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

# Tables owned by LangGraph's PostgresSaver (created by its own setup()). They
# are not part of our metadata, so autogenerate must never propose dropping them.
_EXTERNAL_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


def include_object(object_, name, type_, reflected, compare_to) -> bool:
    if type_ == "table" and name in _EXTERNAL_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade head --sql``)."""
    context.configure(
        url=sync_url().render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(sync_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
