"""Repository for lead database operations."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Lead


class LeadRepository:
    """Database access layer for lead records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ========================================================
    # Create
    # ========================================================

    async def create(
        self,
        *,
        customer_id: int,
        status: str = "new",
        intent: str | None = None,
        budget_min: Decimal | float | int | None = None,
        budget_max: Decimal | float | int | None = None,
        preferred_location: str | None = None,
        preferred_property_type: str | None = None,
        bedrooms_min: int | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Lead:
        """Create and flush a new lead."""

        lead = Lead(
            customer_id=customer_id,
            status=status,
            intent=intent,
            budget_min=budget_min,
            budget_max=budget_max,
            preferred_location=preferred_location,
            preferred_property_type=preferred_property_type,
            bedrooms_min=bedrooms_min,
            notes=notes,
            metadata_json=metadata,
        )

        self.session.add(lead)
        await self.session.flush()

        return lead

    # ========================================================
    # Read
    # ========================================================

    async def get_by_id(self, lead_id: int) -> Lead | None:
        """Return a lead by its database ID."""

        result = await self.session.execute(
            select(Lead).where(Lead.id == lead_id)
        )

        return result.scalar_one_or_none()

    async def list_by_customer(
        self,
        customer_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Lead]:
        """Return leads belonging to a specific customer."""

        result = await self.session.execute(
            select(Lead)
            .where(Lead.customer_id == customer_id)
            .order_by(Lead.id)
            .offset(offset)
            .limit(limit)
        )

        return result.scalars().all()

    async def list_by_status(
        self,
        status: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Lead]:
        """Return leads with the requested status."""

        result = await self.session.execute(
            select(Lead)
            .where(Lead.status == status)
            .order_by(Lead.id)
            .offset(offset)
            .limit(limit)
        )

        return result.scalars().all()

    # ========================================================
    # Update
    # ========================================================

    async def update(
        self,
        lead_id: int,
        **fields: Any,
    ) -> Lead | None:
        """Update mutable fields on an existing lead."""

        lead = await self.get_by_id(lead_id)

        if lead is None:
            return None

        allowed_fields = {
            "status",
            "intent",
            "budget_min",
            "budget_max",
            "preferred_location",
            "preferred_property_type",
            "bedrooms_min",
            "notes",
            "metadata",
        }

        invalid_fields = set(fields) - allowed_fields

        if invalid_fields:
            raise ValueError(
                "Unsupported lead fields: "
                + ", ".join(sorted(invalid_fields))
            )

        for field_name, value in fields.items():
            if field_name == "metadata":
                field_name = "metadata_json"

            setattr(lead, field_name, value)

        await self.session.flush()

        return lead

    # ========================================================
    # Delete
    # ========================================================

    async def delete(self, lead_id: int) -> bool:
        """Delete a lead by its database ID."""

        lead = await self.get_by_id(lead_id)

        if lead is None:
            return False

        await self.session.delete(lead)
        await self.session.flush()

        return True