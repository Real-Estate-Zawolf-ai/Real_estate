"""Customer-relationship management services."""

from app.crm.client import CRMClient
from app.crm.customers import CreateCustomer, Customer, CustomerService
from app.crm.interactions import CreateInteraction, InteractionService
from app.crm.leads import CreateLead, LeadService, LeadStatus

__all__ = [
    "CRMClient",
    "CreateCustomer",
    "Customer",
    "CustomerService",
    "CreateInteraction",
    "InteractionService",
    "CreateLead",
    "LeadService",
    "LeadStatus",
]
