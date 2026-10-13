"""Dialect-aware atomic insert helpers (PostgreSQL in production, SQLite in tests)."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session


def insert_ignore(session: Session, model: type, values: dict[str, Any], index_elements: list[str]) -> bool:
    """``INSERT ... ON CONFLICT DO NOTHING`` on PostgreSQL and SQLite.

    Atomic across workers, so two requests creating the same row never raise.
    Returns True when this call inserted the row, False when it already existed.

    The verdict comes from ``RETURNING``, not ``rowcount``: with psycopg 3 the
    row count of an INSERT is reported as -1, which would read as "inserted"
    every time and silently disable every caller's duplicate protection.
    """
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover - only the two supported databases
        raise RuntimeError(f"Unsupported database dialect: {dialect}")
    statement = (
        insert(model)
        .values(**values)
        .on_conflict_do_nothing(index_elements=index_elements)
        .returning(getattr(model, index_elements[0]))
    )
    return session.execute(statement).first() is not None
