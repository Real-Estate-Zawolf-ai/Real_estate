"""SQLAlchemy ORM models for the real-estate assistant."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ============================================================
# Base
# ============================================================


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


# ============================================================
# Property
# ============================================================


class Property(Base):
    """Real-estate property stored in the application database."""

    __tablename__ = "properties"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # External/source property identifier.
    property_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    property_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    price: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    currency: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    area: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    area_unit: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    bedrooms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    bathrooms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    city: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    district: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    neighborhood: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    compound: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    transaction_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    payment_plan: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    availability: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    finishing: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    delivery: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    amenities: Mapped[list[Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    source: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ============================================================
# Customer
# ============================================================


class Customer(Base):
    """Customer or prospective buyer/renter interacting with the assistant."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    phone: Mapped[str | None] = mapped_column(
        String(50),
        unique=True,
        nullable=True,
        index=True,
    )

    email: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )

    preferred_language: Mapped[str] = mapped_column(
        String(10),
        default="en",
        nullable=False,
    )

    source: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    leads: Mapped[list["Lead"]] = relationship(
        "Lead",
        back_populates="customer",
        cascade="all, delete-orphan",
    )


# ============================================================
# Lead
# ============================================================


class Lead(Base):
    """Sales lead associated with a customer."""

    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="new",
        nullable=False,
        index=True,
    )

    intent: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    budget_min: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    budget_max: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    preferred_location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    preferred_property_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    bedrooms_min: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # SQLAlchemy reserves the attribute name "metadata".
    # Use "metadata_json" in Python while keeping the database
    # column name as "metadata".
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    customer: Mapped["Customer"] = relationship(
        "Customer",
        back_populates="leads",
    )