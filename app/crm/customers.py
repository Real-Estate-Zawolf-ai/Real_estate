"""Customer records and customer-management operations (PostgreSQL)."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.crm.client import CRMClient
from app.database import models as db
from app.helpers.phone import normalize_phone


class CreateCustomer(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    phone_number: str | None = Field(default=None, max_length=30)
    email: str | None = Field(default=None, max_length=320)
    preferred_language: str = Field(default="en", min_length=2, max_length=10)

    @field_validator("full_name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("full_name cannot be blank.")
        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else None


class Customer(CreateCustomer):
    id: UUID = Field(default_factory=uuid4)
    phone_verified: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DuplicateCustomerError(ValueError):
    """Raised when a matching customer contact already exists."""


class CustomerNotFoundError(LookupError):
    """Raised when a customer cannot be found."""


def _entity(row: db.Customer) -> Customer:
    return Customer(
        id=row.public_id,
        full_name=row.name or "",
        phone_number=row.phone,
        email=row.email,
        preferred_language=row.preferred_language,
        phone_verified=row.phone_verified,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class CustomerService:
    def __init__(self, client: CRMClient | None = None) -> None:
        self.client = client or CRMClient()

    def create(self, request: CreateCustomer) -> Customer:
        """Create a customer unless their email or phone already exists."""
        phone = (normalize_phone(request.phone_number) or request.phone_number) if request.phone_number else None
        try:
            with self.client.session() as session:
                if request.email and session.scalar(select(db.Customer.id).where(db.Customer.email == request.email)):
                    raise DuplicateCustomerError("A customer already uses this email.")
                if phone and session.scalar(select(db.Customer.id).where(db.Customer.phone == phone)):
                    raise DuplicateCustomerError("A customer already uses this phone number.")
                row = db.Customer(
                    name=request.full_name,
                    phone=phone,
                    email=request.email,
                    preferred_language=request.preferred_language,
                    source="api",
                )
                session.add(row)
                session.flush()
                session.refresh(row)
                return _entity(row)
        except IntegrityError as error:  # a concurrent request won the race
            raise DuplicateCustomerError("A customer with these contact details already exists.") from error

    def get(self, customer_id: UUID) -> Customer:
        with self.client.session() as session:
            row = session.scalar(select(db.Customer).where(db.Customer.public_id == customer_id))
            if row is None:
                raise CustomerNotFoundError(f"Customer {customer_id} was not found.")
            return _entity(row)
