"""Customer records and customer-management operations."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from app.crm.client import CRMClient


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DuplicateCustomerError(ValueError):
    """Raised when a matching customer contact already exists."""


class CustomerNotFoundError(LookupError):
    """Raised when a customer cannot be found."""


class CustomerService:
    def __init__(self, client: CRMClient) -> None:
        self.client = client

    def create(self, request: CreateCustomer) -> Customer:
        """Create a customer unless their email or phone already exists."""
        for customer in self.client.list("customers"):
            if request.email and request.email == customer.email:
                raise DuplicateCustomerError("A customer already uses this email.")
            if request.phone_number and request.phone_number == customer.phone_number:
                raise DuplicateCustomerError("A customer already uses this phone number.")
        customer = Customer(**request.model_dump())
        return self.client.save("customers", customer.id, customer)

    def get(self, customer_id: UUID) -> Customer:
        customer = self.client.get("customers", customer_id)
        if customer is None:
            raise CustomerNotFoundError(f"Customer {customer_id} was not found.")
        return customer
