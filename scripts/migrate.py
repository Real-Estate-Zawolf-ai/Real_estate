"""Apply database migrations and create the LangGraph checkpoint tables.

    python scripts/migrate.py            # upgrade to the latest revision
    python scripts/migrate.py --check    # exit 1 if the database is behind (no changes)

Idempotent and safe to run on every start. A Neon database that has been idle is
suspended and takes a few seconds to wake, so connection failures are retried.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from app.database.migrations import schema_status  # noqa: E402
from app.database.session import get_engine, is_postgres  # noqa: E402
from app.observability.logging import redact  # noqa: E402


# Errors no amount of waiting fixes: fail at once instead of retrying for a minute.
_PERMANENT = ("password authentication failed", "does not exist", "does not support ssl", "ssl required", "no pg_hba.conf entry")


def wait_for_database(attempts: int = 6) -> None:
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            with get_engine().connect():
                return
        except OperationalError as error:
            first_line = redact(str(error).splitlines()[0])
            if attempt == attempts or any(marker in first_line.lower() for marker in _PERMANENT):
                sys.exit(f"Database unreachable (attempt {attempt}/{attempts}): {first_line}")
            print(f"Database not ready (attempt {attempt}/{attempts}); retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.7, 15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only report whether the schema is up to date")
    args = parser.parse_args()

    if not is_postgres():
        sys.exit("DATABASE_URL (or POSTGRES_*) must point at PostgreSQL to run migrations.")
    wait_for_database()

    if args.check:
        up_to_date, detail = schema_status()
        print(detail)
        sys.exit(0 if up_to_date else 1)

    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    _, detail = schema_status()
    print(f"Database schema {detail}", flush=True)

    from app.agents.checkpointer import close_checkpointer, setup_checkpointer

    setup_checkpointer()
    close_checkpointer()
    print("Conversation-memory tables ready", flush=True)


if __name__ == "__main__":
    main()
