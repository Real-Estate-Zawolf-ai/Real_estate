from __future__ import annotations

from threading import Lock
from uuid import uuid4


class CRMService:

    def __init__(self) -> None:
        self._customers: dict[str, dict] = {}
        self._leads: dict[str, dict] = {}
        self._interactions: list[dict] = []
        self._lock = Lock()

    def get_customer(
        self,
        customer_id: str,
    ) -> dict | None:
        with self._lock:
            customer = self._customers.get(customer_id)
            return dict(customer) if customer else None

    def create_lead(
        self,
        customer: dict,
    ) -> str:
        customer_id = str(customer.get("id") or uuid4())
        lead_id = str(uuid4())
        with self._lock:
            self._customers[customer_id] = {**customer, "id": customer_id}
            self._leads[lead_id] = {"id": lead_id, "customer_id": customer_id, "notes": []}
        return lead_id

    def update_lead(
        self,
        customer_id: str,
        data: dict,
    ) -> None:
        with self._lock:
            matching = [lead for lead in self._leads.values() if lead["customer_id"] == customer_id]
            if not matching:
                raise KeyError(f"No lead found for customer {customer_id}")
            matching[-1].update(data)

    def add_note(
        self,
        customer_id: str,
        note: str,
    ) -> None:
        if not note.strip():
            raise ValueError("note cannot be blank")
        with self._lock:
            matching = [lead for lead in self._leads.values() if lead["customer_id"] == customer_id]
            if not matching:
                raise KeyError(f"No lead found for customer {customer_id}")
            matching[-1].setdefault("notes", []).append(note.strip())

    def record_interaction(
        self,
        *,
        customer_id: str,
        channel: str,
        content: str,
        response: str,
    ) -> None:
        with self._lock:
            self._customers.setdefault(customer_id, {"id": customer_id})
            self._interactions.append(
                {
                    "customer_id": customer_id,
                    "channel": channel,
                    "content": content,
                    "response": response,
                }
            )


crm_service = CRMService()
