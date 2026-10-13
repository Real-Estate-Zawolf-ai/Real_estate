"""Durable conversation memory (LangGraph checkpointer on PostgreSQL).

Every customer turn - website text, website voice, WhatsApp - runs through one
LangGraph workflow keyed by a conversation ``thread_id``. With PostgreSQL behind
it, that state survives restarts and is shared by every API worker.

Three things live here:

* ``build_checkpointer()`` - ``PostgresSaver`` over its own connection pool, or
  an in-process ``MemorySaver`` outside production when no database is set.
* ``ThreadLock`` - serialises turns of one conversation across workers with a
  *transaction-level* advisory lock. (Session-level locks silently break behind
  a transaction-mode pooler such as Neon's ``-pooler`` host; this form does not.)
* ``prune_thread`` / ``purge_stale_threads`` - bound storage. LangGraph writes a
  checkpoint per step plus subgraph checkpoints, each holding the full message
  list, which would exhaust a free database; only the latest state is needed.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Iterator

from langgraph.checkpoint.memory import MemorySaver

from app.database.session import is_postgres
from app.database.url import libpq_conninfo
from app.helpers.config import get_settings


logger = logging.getLogger(__name__)

_LOCK_WAIT = "45s"
_pool_guard = Lock()
_saver: Any = None
_saver_pool: Any = None
_lock_pool: Any = None


def _open_pool(max_size: int, name: str) -> Any:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=libpq_conninfo(),
        min_size=0,
        max_size=max_size,
        kwargs={
            # autocommit: PostgresSaver manages its own transactions and runs
            # CREATE INDEX CONCURRENTLY in setup().
            "autocommit": True,
            "row_factory": dict_row,
            # Safe behind a transaction-mode pooler; harmless otherwise.
            "prepare_threshold": None,
            "connect_timeout": 10,
        },
        # Neon suspends idle computes; discard connections that went stale.
        check=ConnectionPool.check_connection,
        max_idle=240,
        max_lifetime=1800,
        timeout=20,
        name=name,
        open=False,
    )
    pool.open()
    return pool


def build_checkpointer() -> Any:
    """Return the process-wide PostgreSQL checkpointer (created once).

    Without a database (development/tests) each call returns a new, isolated
    in-process ``MemorySaver`` instead.
    """
    global _saver, _saver_pool
    with _pool_guard:
        if _saver is not None:
            return _saver
        if not is_postgres():
            if get_settings().is_production:
                raise RuntimeError("Production requires PostgreSQL for conversation memory")
            # Not cached: every caller gets its own isolated in-process store.
            logger.warning("Conversation memory is in-process (no PostgreSQL configured): lost on restart")
            return MemorySaver()

        from langgraph.checkpoint.postgres import PostgresSaver

        _saver_pool = _open_pool(max_size=6, name="aqar-checkpointer")
        _saver = PostgresSaver(_saver_pool)
        logger.info("Conversation memory: PostgreSQL checkpointer")
        return _saver


def setup_checkpointer() -> None:
    """Create/upgrade the checkpoint tables. Idempotent; safe to run on every start."""
    saver = build_checkpointer()
    if hasattr(saver, "setup") and _saver_pool is not None:
        with _saver_pool.connection() as connection:
            # One worker migrates at a time.
            connection.execute("SELECT pg_advisory_lock(%s)", (_lock_key("checkpoint-setup"),))
            try:
                saver.setup()
            finally:
                connection.execute("SELECT pg_advisory_unlock(%s)", (_lock_key("checkpoint-setup"),))


def close_checkpointer() -> None:
    global _saver, _saver_pool, _lock_pool
    with _pool_guard:
        for pool in (_saver_pool, _lock_pool):
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    logger.exception("Error closing a connection pool")
        _saver = _saver_pool = _lock_pool = None


def checkpointer_ready() -> tuple[bool, str | None]:
    """Probe for /health/ready."""
    try:
        saver = build_checkpointer()
        if _saver_pool is None:
            return True, None
        with _saver_pool.connection() as connection:
            connection.execute("SELECT 1 FROM checkpoints LIMIT 1")
        return saver is not None, None
    except Exception as error:
        return False, type(error).__name__


# ---------------------------------------------------------------------------
# Cross-worker turn lock
# ---------------------------------------------------------------------------


def _lock_key(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big", signed=True)


@contextmanager
def thread_lock(thread_id: str) -> Iterator[None]:
    """Hold the conversation's lock for the duration of one turn.

    Only active with PostgreSQL. Held in an open transaction on a dedicated
    connection, released automatically on commit or if the worker dies.
    """
    global _lock_pool
    if not is_postgres():
        yield
        return
    with _pool_guard:
        if _lock_pool is None:
            _lock_pool = _open_pool(max_size=4, name="aqar-turn-locks")
    with _lock_pool.connection() as connection:
        with connection.transaction():
            connection.execute(f"SET LOCAL lock_timeout = '{_LOCK_WAIT}'")
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_lock_key(f"thread:{thread_id}"),))
            yield


# ---------------------------------------------------------------------------
# Storage bounds
# ---------------------------------------------------------------------------

_PRUNE_STATEMENTS = (
    # Subgraph runs write a throw-away checkpoint under a fresh namespace each
    # turn; once the turn is over they are dead weight.
    "DELETE FROM checkpoints WHERE thread_id = %(t)s AND checkpoint_ns <> ''",
    # Keep only the newest top-level checkpoint (ids sort chronologically).
    """
    DELETE FROM checkpoints
    WHERE thread_id = %(t)s AND checkpoint_ns = ''
      AND checkpoint_id < (
        SELECT max(checkpoint_id) FROM checkpoints WHERE thread_id = %(t)s AND checkpoint_ns = ''
      )
    """,
    # Pending writes belonging to checkpoints that no longer exist.
    """
    DELETE FROM checkpoint_writes w
    WHERE w.thread_id = %(t)s AND NOT EXISTS (
        SELECT 1 FROM checkpoints c
        WHERE c.thread_id = w.thread_id AND c.checkpoint_ns = w.checkpoint_ns AND c.checkpoint_id = w.checkpoint_id
    )
    """,
    # Channel values no surviving checkpoint refers to.
    """
    DELETE FROM checkpoint_blobs b
    WHERE b.thread_id = %(t)s AND NOT EXISTS (
        SELECT 1 FROM checkpoints c, jsonb_each_text(c.checkpoint -> 'channel_versions') v
        WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns
          AND v.key = b.channel AND v.value = b.version
    )
    """,
)


def prune_thread(thread_id: str) -> None:
    """Drop everything but the latest state of one conversation. Best effort."""
    if _saver_pool is None:
        return
    try:
        with _saver_pool.connection() as connection:
            for statement in _PRUNE_STATEMENTS:
                connection.execute(statement, {"t": thread_id})
    except Exception:
        logger.exception("Could not prune conversation checkpoints | thread=%s", thread_id[:16])


def purge_stale_threads(older_than_days: int = 90) -> int:
    """Delete conversations idle for ``older_than_days``; returns how many."""
    if _saver_pool is None or not hasattr(_saver, "delete_thread"):
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    with _saver_pool.connection() as connection:
        # checkpoint ids are UUIDv6 (time-ordered); the newest id marks last activity.
        rows = connection.execute(
            """
            SELECT thread_id FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            HAVING max((checkpoint ->> 'ts')::timestamptz) < %(cutoff)s
            """,
            {"cutoff": cutoff},
        ).fetchall()
    for row in rows:
        _saver.delete_thread(row["thread_id"])
    return len(rows)
