from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


Channel = Literal[
    "website",
    "whatsapp",
]

Language = Literal[
    "ar",
    "en",
]


class ConversationState(TypedDict, total=False):


    messages: Annotated[list[BaseMessage], add_messages]

    query: str
    rewritten_query: str

    channel: Channel
    language: Language

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    intent: str
    next_agent: str

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer_id: str

    # --------------------------------------------------------
    # Property
    # --------------------------------------------------------

    requirements: dict
    candidates: list[dict]
    cited_property_ids: list[str]
    grounding_property_ids: list[str]
    selected_property: dict | None

    # --------------------------------------------------------
    # Qualification
    # --------------------------------------------------------

    qualification: dict

    # --------------------------------------------------------
    # Scheduling
    # --------------------------------------------------------

    appointment: dict | None

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response: str

_PROPERTY_RE = re.compile(
    r"\b(?:properties|property|real estate|apartments?|flats?|villas?|houses?|homes?|"
    r"chalets?|commercial(?: unit| property)?)\b|عقار(?:ات)?|شقه?|شقق|فيلا|فلل?|"
    r"بيت|بيوت|منزل|منازل|"
    r"شاليهات?|وحدات تجاريه|وحده تجاريه|تجاري"
)
_SCHEDULING_RE = re.compile(
    r"\b(?:schedule|appointment|viewing|book (?:a )?(?:visit|viewing))\b|"
    r"موعد|مواعيد|معاينه|معاينات|احجز|حجز"
)
_QUALIFICATION_RE = re.compile(
    r"\b(?:budget|timeline|buy|buying|rent|renting|purchase)\b|"
    r"\b(?:under|below|less than|up to|over|above|more than)\s+\d|"
    r"(?:اقل من|حد اقصي|حتى|فوق|اكثر من)\s*\d|"
    r"ميزانية|شراء|ايجار|استئجار"
)
_REFINEMENT_RE = re.compile(
    r"\b\d{1,2}\s*(?:\+\s*)?(?:bedrooms?|beds?|rooms?)\b|"
    r"\d{1,2}\s*(?:غرف|غرفه|غرفة|نوم)|"
    r"\b(?:under|below|less than|up to|over|above|more than)\s+\d|"
    r"(?:اقل من|حد اقصي|حتى|فوق|اكثر من)\s*\d|"
    r"\b\d+(?:[.,]\d+)?\s*(?:million|millions|m|k)\b|"
    r"\b(?:rentals?|for rent|for sale|buy|rent)\b|ايجار|استئجار|للبيع|للايجار|للإيجار|"
    r"\b(?:new cairo|sheikh zayed|north coast)\b|القاهرة الجديدة|الشيخ زايد|الساحل الشمالي"
)
_ACKS = {
    "تمام", "شكرا", "شكرا جزيلا", "ماشي", "حلو", "اوكي", "ok", "okay", "thanks",
    "thank you", "great", "hi", "hello", "اهلا", "اهلا وسهلا", "ازيك", "عامل ايه",
}
_UNRELATED_RE = re.compile(
    r"(?:\bweather\b|\b(?:tell me|say) a joke\b|\bhow are you\b|\bwhat s the news\b|"
    r"\bwhat s up\b|الجو|اخبار|الاخبار|عامل ايه|عامله ايه|ازيك)"
)


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
        **dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789")),
    }))
    value = re.sub(r"[\u064B-\u065F\u0670ـ]", "", value)
    return re.sub(r"[^\w\s]", " ", value).strip()


def _message_role_content(message: object) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", message.get("type", ""))).lower()
        content = message.get("content", "")
    else:
        role = str(getattr(message, "type", "")).lower()
        content = getattr(message, "content", "")
    if role in {"human", "user"}:
        role = "user"
    if isinstance(content, str):
        return role, _normalize_text(content)
    return role, _normalize_text(str(content))


def _has_property_context(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages[-12:]):
        role, content = _message_role_content(message)
        if role == "user" and _PROPERTY_RE.search(content):
            return True
    return False


def router(state: ConversationState) -> dict:
    """Route new property requests and contextual refinements to property search."""
    query = _normalize_text(state.get("query", ""))
    messages = state.get("messages", [])

    if _PROPERTY_RE.search(query):
        intent = "property_search"
    elif _SCHEDULING_RE.search(query):
        intent = "scheduling"
    elif query in _ACKS or _UNRELATED_RE.search(query):
        intent = "conversational"
    elif _REFINEMENT_RE.search(query) and _has_property_context(messages):
        intent = "property_search"
    elif _QUALIFICATION_RE.search(query):
        intent = "qualification"
    else:
        intent = "conversational"
    return {"intent": intent}

def route_agent(state: ConversationState) -> str:


    intent = state.get("intent")

    if intent == "property_search":
        return "property"

    if intent == "qualification":
        return "qualification"

    if intent == "scheduling":
        return "scheduling"

    return "conversational"
