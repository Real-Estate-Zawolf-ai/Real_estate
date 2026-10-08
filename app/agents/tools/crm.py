from __future__ import annotations

from threading import Lock
from uuid import uuid4


class CRMService:

    def __init__(self) -> None:
        self._customers: dict[str, dict] = {}
        self._leads: dict[str, dict] = {}
        self._interactions: list[dict] = []
        self._handoffs: list[dict] = []
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

    # --------------------------------------------------------
    # Agent-facing helpers (one open lead per customer)
    # --------------------------------------------------------

    def upsert_customer(self, customer_id: str, **fields: object) -> dict:
        clean = {key: value for key, value in fields.items() if value not in (None, "")}
        with self._lock:
            customer = self._customers.setdefault(customer_id, {"id": customer_id})
            customer.update(clean)
            return dict(customer)

    def ensure_lead(self, customer_id: str, *, source: str) -> str:
        with self._lock:
            self._customers.setdefault(customer_id, {"id": customer_id})
            for lead in self._leads.values():
                if lead["customer_id"] == customer_id:
                    return lead["id"]
            lead_id = str(uuid4())
            self._leads[lead_id] = {
                "id": lead_id,
                "customer_id": customer_id,
                "source": source,
                "status": "new",
                "notes": [],
            }
            return lead_id

    def get_lead(self, customer_id: str) -> dict | None:
        with self._lock:
            matching = [lead for lead in self._leads.values() if lead["customer_id"] == customer_id]
            return dict(matching[-1]) if matching else None

    def record_handoff(self, customer_id: str, *, source: str, reason: str, priority: str, summary: dict) -> dict:
        lead_id = self.ensure_lead(customer_id, source=source)
        handoff = {"id": str(uuid4()), "lead_id": lead_id, "customer_id": customer_id,
                   "reason": reason, "priority": priority, "summary": summary, "status": "open"}
        with self._lock:
            self._handoffs.append(handoff)
            self._leads[lead_id]["status"] = "handoff_requested"
        return dict(handoff)

    def list_handoffs(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._handoffs]


crm_service = CRMService()
