"""production persistence

Adds the tables that replace the in-memory CRM, interaction log, handoff queue
and booking calendar, plus the cross-worker replay guard (webhook ids, voice
token nonces), the staff-notification outbox and the admin audit trail.

PostgreSQL only: the booking table gets an exclusion constraint so two
overlapping active viewings for one property can never be stored, no matter
which worker or request writes them.

Revision ID: a7c1e0d2b9f4
Revises: ce49f8c2929c
Create Date: 2026-10-10 19:15:49

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c1e0d2b9f4"
down_revision: Union[str, None] = "ce49f8c2929c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ts(name: str, *, nullable: bool = False, default_now: bool = False) -> sa.Column:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.text("now()") if default_now else None,
        nullable=nullable,
    )


def upgrade() -> None:
    # --- existing tables: new columns -----------------------------------------
    # public_id is NOT NULL + unique, so add it nullable, backfill, then tighten.
    for table in ("customers", "leads"):
        op.add_column(table, sa.Column("public_id", sa.Uuid(), nullable=True))
        op.execute(f"UPDATE {table} SET public_id = gen_random_uuid() WHERE public_id IS NULL")
        op.alter_column(table, "public_id", nullable=False)
        op.create_unique_constraint(f"{table}_public_id_key", table, ["public_id"])

    op.add_column("customers", sa.Column("phone_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False))
    op.add_column("customers", sa.Column("external_id", sa.String(length=120), nullable=True))
    op.create_index(op.f("ix_customers_external_id"), "customers", ["external_id"], unique=True)
    op.add_column("customers", _ts("phone_verified_at", nullable=True))
    op.add_column("customers", sa.Column("phone_verification_method", sa.String(length=30), nullable=True))

    op.add_column("leads", sa.Column("source", sa.String(length=50), nullable=True))
    op.add_column("leads", sa.Column("score", sa.Integer(), server_default=sa.text("0"), nullable=False))
    op.add_column("leads", sa.Column("property_ref", sa.String(length=100), nullable=True))

    # --- customers: phone/email identify a person, not a conversation --------
    op.drop_index(op.f("ix_customers_phone"), table_name="customers")
    op.create_index(op.f("ix_customers_phone"), "customers", ["phone"], unique=False)
    op.drop_index(op.f("ix_customers_email"), table_name="customers")
    op.create_index(op.f("ix_customers_email"), "customers", ["email"], unique=False)

    # --- interactions ---------------------------------------------------------
    op.create_table(
        "interactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("lead_id", sa.Integer(), nullable=True),
        sa.Column("channel", sa.String(length=50), nullable=False),
        sa.Column("direction", sa.String(length=10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=True),
        _ts("created_at", default_now=True),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["lead_id"], ["leads.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(op.f("ix_interactions_customer_id"), "interactions", ["customer_id"])
    op.create_index(op.f("ix_interactions_created_at"), "interactions", ["created_at"])

    # --- handoffs -------------------------------------------------------------
    op.create_table(
        "handoffs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("lead_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("reason", sa.String(length=100), nullable=False),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("claimed_by", sa.String(length=100), nullable=True),
        _ts("resolved_at", nullable=True),
        _ts("created_at", default_now=True),
        _ts("updated_at", default_now=True),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["lead_id"], ["leads.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(op.f("ix_handoffs_customer_id"), "handoffs", ["customer_id"])
    op.create_index(op.f("ix_handoffs_status"), "handoffs", ["status"])

    # --- bookings -------------------------------------------------------------
    op.create_table(
        "bookings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.Uuid(), nullable=False),
        sa.Column("property_id", sa.String(length=100), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("lead_id", sa.Integer(), nullable=True),
        sa.Column("agent_id", sa.String(length=100), nullable=True),
        _ts("starts_at"),
        _ts("ends_at"),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("contact_phone", sa.String(length=30), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        _ts("hold_expires_at", nullable=True),
        _ts("confirmed_at", nullable=True),
        sa.Column("confirmed_by", sa.String(length=100), nullable=True),
        _ts("cancelled_at", nullable=True),
        sa.Column("cancelled_by", sa.String(length=20), nullable=True),
        sa.Column("cancel_reason", sa.String(length=255), nullable=True),
        _ts("created_at", default_now=True),
        _ts("updated_at", default_now=True),
        sa.CheckConstraint("ends_at > starts_at", name="ck_bookings_interval"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["lead_id"], ["leads.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(op.f("ix_bookings_customer_id"), "bookings", ["customer_id"])
    op.create_index(op.f("ix_bookings_status"), "bookings", ["status"])
    op.create_index("ix_bookings_property_starts", "bookings", ["property_id", "starts_at"])

    # No two active viewings of one property may overlap in time. Enforced by the
    # database so concurrent workers cannot double-book a slot.
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
    op.execute(
        """
        ALTER TABLE bookings ADD CONSTRAINT bookings_no_overlap
        EXCLUDE USING gist (
            property_id WITH =,
            tstzrange(starts_at, ends_at, '[)') WITH &&
        ) WHERE (status IN ('pending_confirmation', 'confirmed'))
        """
    )

    # --- staff notification outbox -------------------------------------------
    op.create_table(
        "staff_notifications",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kind", sa.String(length=50), nullable=False),
        sa.Column("ref_type", sa.String(length=30), nullable=True),
        sa.Column("ref_id", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        _ts("created_at", default_now=True),
        _ts("sent_at", nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_staff_notifications_status"), "staff_notifications", ["status"])

    # --- replay guard ---------------------------------------------------------
    op.create_table(
        "replay_guard",
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        _ts("expires_at"),
        _ts("created_at", default_now=True),
        sa.PrimaryKeyConstraint("kind", "key"),
    )
    op.create_index(op.f("ix_replay_guard_expires_at"), "replay_guard", ["expires_at"])

    # --- admin audit ----------------------------------------------------------
    op.create_table(
        "admin_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("action", sa.String(length=60), nullable=False),
        sa.Column("target", sa.String(length=100), nullable=True),
        sa.Column("actor", sa.String(length=100), nullable=False),
        sa.Column("client", sa.String(length=64), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        _ts("created_at", default_now=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_admin_audit_created_at"), "admin_audit", ["created_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_admin_audit_created_at"), table_name="admin_audit")
    op.drop_table("admin_audit")
    op.drop_index(op.f("ix_replay_guard_expires_at"), table_name="replay_guard")
    op.drop_table("replay_guard")
    op.drop_index(op.f("ix_staff_notifications_status"), table_name="staff_notifications")
    op.drop_table("staff_notifications")

    op.execute("ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_no_overlap")
    op.drop_index("ix_bookings_property_starts", table_name="bookings")
    op.drop_index(op.f("ix_bookings_status"), table_name="bookings")
    op.drop_index(op.f("ix_bookings_customer_id"), table_name="bookings")
    op.drop_table("bookings")

    op.drop_index(op.f("ix_handoffs_status"), table_name="handoffs")
    op.drop_index(op.f("ix_handoffs_customer_id"), table_name="handoffs")
    op.drop_table("handoffs")

    op.drop_index(op.f("ix_interactions_created_at"), table_name="interactions")
    op.drop_index(op.f("ix_interactions_customer_id"), table_name="interactions")
    op.drop_table("interactions")

    op.drop_index(op.f("ix_customers_email"), table_name="customers")
    op.create_index(op.f("ix_customers_email"), "customers", ["email"], unique=True)
    op.drop_index(op.f("ix_customers_phone"), table_name="customers")
    op.create_index(op.f("ix_customers_phone"), "customers", ["phone"], unique=True)

    op.drop_column("leads", "property_ref")
    op.drop_column("leads", "score")
    op.drop_column("leads", "source")
    op.drop_column("customers", "phone_verification_method")
    op.drop_column("customers", "phone_verified_at")
    op.drop_index(op.f("ix_customers_external_id"), table_name="customers")
    op.drop_column("customers", "external_id")
    op.drop_column("customers", "phone_verified")
    for table in ("leads", "customers"):
        op.drop_constraint(f"{table}_public_id_key", table, type_="unique")
        op.drop_column(table, "public_id")
