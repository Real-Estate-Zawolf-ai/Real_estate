"""Schema-version checks, so the API never runs against a stale database."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from app.database.session import get_engine, is_postgres


_ROOT = Path(__file__).resolve().parents[2]


def head_revision() -> str | None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(_ROOT / "alembic"))
    return ScriptDirectory.from_config(config).get_current_head()


def current_revision() -> str | None:
    with get_engine().connect() as connection:
        try:
            return connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
        except Exception:
            return None


def schema_status() -> tuple[bool, str]:
    """(up_to_date, human-readable detail). SQLite dev databases are self-created."""
    if not is_postgres():
        return True, "sqlite (created from models)"
    head, current = head_revision(), current_revision()
    if current is None:
        return False, "no migrations applied: run `alembic upgrade head`"
    if current != head:
        return False, f"database at {current}, code expects {head}: run `alembic upgrade head`"
    return True, f"at {head}"
