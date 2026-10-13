"""Housekeeping for the free-tier database. Safe to run any time; run it weekly.

    python scripts/maintenance.py                  # dry run: report what would be removed
    python scripts/maintenance.py --apply          # do it
    python scripts/maintenance.py --apply --conversation-days 60 --interaction-days 365

Neon's free plan has 0.5 GB of storage, so unbounded history eventually stops
the app. This removes, in order of how safe it is:

* expired replay-guard rows (webhook ids, spent voice tokens)
* conversation memory (LangGraph checkpoints) idle for ``--conversation-days``
* delivered staff alerts older than ``--alert-days``
* chat transcripts older than ``--interaction-days`` (leads, customers and
  bookings are kept: they are the business record)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402

from app.database import models as db  # noqa: E402
from app.database.session import is_postgres, session_scope  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="delete (default is a dry run)")
    parser.add_argument("--conversation-days", type=int, default=90)
    parser.add_argument("--alert-days", type=int, default=60)
    parser.add_argument("--interaction-days", type=int, default=365)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    verb = "removed" if args.apply else "would remove"

    with session_scope() as session:
        expired = session.scalar(select(func.count()).select_from(db.ReplayGuard).where(db.ReplayGuard.expires_at < now))
        alerts = session.scalar(
            select(func.count()).select_from(db.StaffNotification).where(
                db.StaffNotification.status.in_(("sent", "abandoned")),
                db.StaffNotification.created_at < now - timedelta(days=args.alert_days),
            )
        )
        interactions = session.scalar(
            select(func.count()).select_from(db.Interaction).where(db.Interaction.created_at < now - timedelta(days=args.interaction_days))
        )
        if args.apply:
            session.execute(delete(db.ReplayGuard).where(db.ReplayGuard.expires_at < now))
            session.execute(
                delete(db.StaffNotification).where(
                    db.StaffNotification.status.in_(("sent", "abandoned")),
                    db.StaffNotification.created_at < now - timedelta(days=args.alert_days),
                )
            )
            session.execute(delete(db.Interaction).where(db.Interaction.created_at < now - timedelta(days=args.interaction_days)))
    print(f"replay-guard rows: {verb} {expired}")
    print(f"delivered staff alerts older than {args.alert_days} days: {verb} {alerts}")
    print(f"chat transcripts older than {args.interaction_days} days: {verb} {interactions}")

    if is_postgres():
        from app.agents.checkpointer import close_checkpointer, purge_stale_threads, setup_checkpointer

        setup_checkpointer()
        if args.apply:
            print(f"conversations idle for {args.conversation_days}+ days: removed {purge_stale_threads(args.conversation_days)}")
        else:
            print(f"conversations idle for {args.conversation_days}+ days: (run with --apply to count and remove)")
        close_checkpointer()
    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
