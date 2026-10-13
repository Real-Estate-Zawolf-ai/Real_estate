"""Cross-worker "first one wins" claims backed by the ``replay_guard`` table.

Used for webhook de-duplication (providers deliver at-least-once) and for
single-use voice tokens. Because the claim is one atomic INSERT in PostgreSQL it
holds across every worker and survives restarts, unlike an in-memory set.

A claim moves ``processing`` -> ``done`` or ``failed``. A ``failed`` claim, or a
``processing`` one older than ``stale_after`` (its worker died), can be taken
over so a provider retry still gets handled instead of being dropped.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, update

from app.database import models as db
from app.database.session import session_scope
from app.database.upsert import insert_ignore


logger = logging.getLogger(__name__)


def claim(kind: str, key: str, *, ttl_seconds: int, stale_after_seconds: int | None = None) -> bool:
    """Return True if the caller now owns ``(kind, key)`` and should process it."""
    now = datetime.now(timezone.utc)
    key = key[:200]
    with session_scope() as session:
        # An expired record no longer protects against anything; clear it so the
        # same id (e.g. a very late retry) can be claimed again.
        session.execute(
            delete(db.ReplayGuard).where(
                db.ReplayGuard.kind == kind, db.ReplayGuard.key == key, db.ReplayGuard.expires_at < now
            )
        )
        inserted = insert_ignore(
            session,
            db.ReplayGuard,
            {"kind": kind, "key": key, "status": "processing", "expires_at": now + timedelta(seconds=ttl_seconds)},
            ["kind", "key"],
        )
        if inserted:
            return True

        retake = db.ReplayGuard.status == "failed"
        if stale_after_seconds is not None:
            retake = retake | (
                (db.ReplayGuard.status == "processing")
                & (db.ReplayGuard.created_at < now - timedelta(seconds=stale_after_seconds))
            )
        taken = session.execute(
            update(db.ReplayGuard)
            .where(db.ReplayGuard.kind == kind, db.ReplayGuard.key == key, retake)
            .values(status="processing", created_at=now, expires_at=now + timedelta(seconds=ttl_seconds))
            .returning(db.ReplayGuard.key)
        )
        return taken.first() is not None  # RETURNING, not rowcount (see insert_ignore)


def settle(kind: str, key: str, *, ok: bool) -> None:
    """Record the outcome of a claimed item (``done`` keeps blocking duplicates)."""
    try:
        with session_scope() as session:
            session.execute(
                update(db.ReplayGuard)
                .where(db.ReplayGuard.kind == kind, db.ReplayGuard.key == key[:200])
                .values(status="done" if ok else "failed")
            )
    except Exception:
        logger.exception("Could not settle replay guard | kind=%s", kind)


def purge_expired() -> int:
    """Delete expired claims; run periodically to keep the table small."""
    with session_scope() as session:
        result = session.execute(delete(db.ReplayGuard).where(db.ReplayGuard.expires_at < datetime.now(timezone.utc)))
        return result.rowcount or 0
