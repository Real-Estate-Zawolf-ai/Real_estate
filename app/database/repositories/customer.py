"""Repository for customer database operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Customer


class CustomerRepository:
    """Database access layer for customers."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ========================================================
    # Create
    # ========================================================

    async def create(
        self,
        *,
        name: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        preferred_language: str = "en",
        source: str | None = None,
        notes: str | None = None,
    ) -> Customer:
        """Create and flush a new customer record."""

        customer = Customer(
            name=name,
            phone=phone,
            email=email,
            preferred_language=preferred_language,
            source=source,
            notes=notes,
        )

        self.session.add(customer)
        await self.session.flush()

        return customer

    # ========================================================
    # Read
    # ========================================================

    async def get_by_id(self, customer_id: int) -> Customer | None:
        """Return a customer by internal database ID."""

        statement = select(Customer).where(
            Customer.id == customer_id
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_phone(self, phone: str) -> Customer | None:
        """Return a customer by phone number."""

        statement = select(Customer).where(
            Customer.phone == phone
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Customer | None:
        """Return a customer by email address."""

        statement = select(Customer).where(
            Customer.email == email
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def list_customers(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Customer]:
        """Return customers using pagination."""

        statement = (
            select(Customer)
            .order_by(Customer.id)
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
        customer_id: int,
        **fields: Any,
    ) -> Customer | None:
        """
        Update an existing customer.

        Only fields defined as customer attributes are accepted.
        The method does not commit the transaction.
        """

        customer = await self.get_by_id(customer_id)

        if customer is None:
            return None

        allowed_fields = {
            "name",
            "phone",
            "email",
            "preferred_language",
            "source",
            "notes",
        }

        invalid_fields = set(fields) - allowed_fields

        if invalid_fields:
            raise ValueError(
                "Unsupported customer fields: "
                + ", ".join(sorted(invalid_fields))
            )

        for field_name, value in fields.items():
            setattr(customer, field_name, value)

        await self.session.flush()

        return customer

    # ========================================================
    # Delete
    # ========================================================

    async def delete(self, customer_id: int) -> bool:
        """
        Permanently delete a customer.

        Related leads are deleted through the configured
        relationship cascade.
        """

        customer = await self.get_by_id(customer_id)

        if customer is None:
            return False

        await self.session.delete(customer)
        await self.session.flush()

        return True