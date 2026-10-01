"""Repository for property database operations."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Property


class PropertyRepository:
    """Database access layer for real-estate properties."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ========================================================
    # Create
    # ========================================================

    async def create(
        self,
        *,
        property_id: str,
        property_type: str | None = None,
        price: Decimal | float | int | None = None,
        currency: str | None = None,
        area: Decimal | float | int | None = None,
        area_unit: str | None = None,
        bedrooms: int | None = None,
        bathrooms: int | None = None,
        location: str | None = None,
        city: str | None = None,
        district: str | None = None,
        neighborhood: str | None = None,
        compound: str | None = None,
        transaction_type: str | None = None,
        payment_plan: str | None = None,
        availability: str | None = None,
        finishing: str | None = None,
        delivery: str | None = None,
        amenities: list[Any] | None = None,
        source: str | None = None,
        url: str | None = None,
        is_active: bool = True,
    ) -> Property:
        """Create and flush a new property record."""

        property_record = Property(
            property_id=property_id,
            property_type=property_type,
            price=price,
            currency=currency,
            area=area,
            area_unit=area_unit,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            location=location,
            city=city,
            district=district,
            neighborhood=neighborhood,
            compound=compound,
            transaction_type=transaction_type,
            payment_plan=payment_plan,
            availability=availability,
            finishing=finishing,
            delivery=delivery,
            amenities=amenities,
            source=source,
            url=url,
            is_active=is_active,
        )

        self.session.add(property_record)
        await self.session.flush()

        return property_record

    # ========================================================
    # Read
    # ========================================================

    async def get_by_id(self, property_id: str) -> Property | None:
        """Return a property by its external property identifier."""

        statement = select(Property).where(
            Property.property_id == property_id
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_database_id(
        self,
        database_id: int,
    ) -> Property | None:
        """Return a property by its internal database ID."""

        statement = select(Property).where(
            Property.id == database_id
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def list_active(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Property]:
        """Return active properties using pagination."""

        statement = (
            select(Property)
            .where(Property.is_active.is_(True))
            .order_by(Property.id)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return result.scalars().all()

    async def list_by_source(
        self,
        source: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Property]:
        """Return properties originating from a specific source."""

        statement = (
            select(Property)
            .where(Property.source == source)
            .order_by(Property.id)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return result.scalars().all()

    # ========================================================
    # Update
    # ========================================================

    async def update(
        self,
        property_id: str,
        **fields: Any,
    ) -> Property | None:
        """
        Update an existing property.

        Only attributes defined on the Property model are accepted.
        The method does not commit the transaction.
        """

        property_record = await self.get_by_id(property_id)

        if property_record is None:
            return None

        allowed_fields = {
            "property_type",
            "price",
            "currency",
            "area",
            "area_unit",
            "bedrooms",
            "bathrooms",
            "location",
            "city",
            "district",
            "neighborhood",
            "compound",
            "transaction_type",
            "payment_plan",
            "availability",
            "finishing",
            "delivery",
            "amenities",
            "source",
            "url",
            "is_active",
        }

        invalid_fields = set(fields) - allowed_fields

        if invalid_fields:
            raise ValueError(
                "Unsupported property fields: "
                + ", ".join(sorted(invalid_fields))
            )

        for field_name, value in fields.items():
            setattr(property_record, field_name, value)

        await self.session.flush()

        return property_record

    # ========================================================
    # Delete / Deactivate
    # ========================================================

    async def deactivate(
        self,
        property_id: str,
    ) -> Property | None:
        """Mark a property as inactive without deleting its record."""

        property_record = await self.get_by_id(property_id)

        if property_record is None:
            return None

        property_record.is_active = False

        await self.session.flush()

        return property_record

    async def delete(self, property_id: str) -> bool:
        """
        Permanently delete a property.

        This method does not commit the transaction.
        """

        property_record = await self.get_by_id(property_id)

        if property_record is None:
            return False

        await self.session.delete(property_record)
        await self.session.flush()

        return True