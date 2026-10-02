from app.crm import CRMClient, CreateCustomer, CreateInteraction, CreateLead
from app.crm import CustomerService, InteractionService, LeadService
from app.crm.interactions import InteractionChannel


def test_customer_lead_and_interaction_workflow():
    client = CRMClient()
    customer = CustomerService(client).create(CreateCustomer(full_name="Nora Ali", email="NORA@example.com"))
    lead = LeadService(client).create(CreateLead(customer_id=customer.id, budget_max=5_000_000))
    interaction = InteractionService(client).record(
        CreateInteraction(
            customer_id=customer.id,
            lead_id=lead.id,
            channel=InteractionChannel.WEBSITE,
            content="Looking for an apartment",
        )
    )
    assert customer.email == "nora@example.com"
    assert interaction.lead_id == lead.id
