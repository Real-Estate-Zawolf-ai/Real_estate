"""Build SQLAlchemy connection URLs from application settings.

One source of truth for the sync engine (services, LangGraph checkpointer,
Alembic) and the legacy async engine. Handles the details hosted PostgreSQL
(Neon) needs: percent-escaped credentials, ``postgres://`` URLs, mandatory TLS
for non-local hosts and driver-specific SSL parameters.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import URL, make_url

from app.helpers.config import Settings, get_settings


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres", "host.docker.internal"}
_DEV_SQLITE = Path(__file__).resolve().parents[2] / "data" / "dev.sqlite3"


class DatabaseConfigError(RuntimeError):
    """The database is required but not (fully) configured."""


def _from_parts(settings: Settings) -> URL | None:
    if not settings.POSTGRES_HOST:
        return None
    missing = [
        name
        for name, value in (
            ("POSTGRES_USERNAME", settings.POSTGRES_USERNAME),
            ("POSTGRES_PASSWORD", settings.POSTGRES_PASSWORD),
            ("POSTGRES_PORT", settings.POSTGRES_PORT),
            ("POSTGRES_MAIN_DATABASE", settings.POSTGRES_MAIN_DATABASE),
        )
        if not value
    ]
    if missing:
        raise DatabaseConfigError("Missing PostgreSQL configuration: " + ", ".join(missing))
    # URL.create escapes special characters in the password.
    return URL.create(
        "postgresql",
        username=settings.POSTGRES_USERNAME,
        password=settings.POSTGRES_PASSWORD,
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_MAIN_DATABASE,
    )


def configured_url(settings: Settings | None = None) -> URL | None:
    """The configured database URL, or None when nothing is configured."""
    settings = settings or get_settings()
    if settings.DATABASE_URL:
        try:
            return make_url(settings.DATABASE_URL)
        except Exception as error:  # ArgumentError carries the raw URL; do not echo it
            raise DatabaseConfigError("DATABASE_URL is not a valid database URL") from error
    return _from_parts(settings)


def resolve_url(settings: Settings | None = None) -> URL:
    """Return the URL to use, falling back to a local SQLite file outside production."""
    settings = settings or get_settings()
    url = configured_url(settings)
    if url is not None:
        return url
    if settings.is_production:
        raise DatabaseConfigError("DATABASE_URL (or POSTGRES_*) must be set in production")
    _DEV_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    return URL.create("sqlite", database=str(_DEV_SQLITE))


def is_sqlite(url: URL) -> bool:
    return url.get_backend_name() == "sqlite"


def database_configured(settings: Settings | None = None) -> bool:
    return configured_url(settings) is not None


def sync_url(settings: Settings | None = None) -> URL:
    """URL for the synchronous psycopg (v3) driver."""
    url = resolve_url(settings)
    if is_sqlite(url):
        return url
    query = dict(url.query)
    if url.host not in _LOCAL_HOSTS:
        query.setdefault("sslmode", "require")
    return url.set(drivername="postgresql+psycopg", query=query)


def async_url(settings: Settings | None = None) -> URL:
    """URL for the asyncpg driver (legacy async repositories)."""
    url = resolve_url(settings)
    if is_sqlite(url):
        raise DatabaseConfigError("The async engine requires PostgreSQL")
    query = dict(url.query)
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    if "ssl" not in query:
        if sslmode and sslmode != "disable":
            query["ssl"] = sslmode
        elif url.host not in _LOCAL_HOSTS:
            query["ssl"] = "require"
    return url.set(drivername="postgresql+asyncpg", query=query)


def libpq_conninfo(settings: Settings | None = None) -> str:
    """Connection string for plain psycopg (LangGraph checkpointer, advisory locks)."""
    url = sync_url(settings)
    if is_sqlite(url):
        raise DatabaseConfigError("A PostgreSQL connection string was requested but the database is SQLite")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def redacted(url: URL) -> str:
    """Safe-to-log form of a URL (password masked)."""
    return url.render_as_string(hide_password=True)
