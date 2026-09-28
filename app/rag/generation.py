"""
generation.py

Final step of the RAG pipeline.

Takes the customer's query plus reranked property evidence and generates
a grounded response.

RAG flow:

    Metadata Filtering
            ↓
    Hybrid Search
            ↓
    Reranking
            ↓
    Generation
            ↓
    Grounded Customer Response

Design principles:

- The LLM must not invent property IDs, prices, availability, payment
  plans, locations, or other property facts.
- Generation never performs retrieval itself.
- Only retrieved SearchResult objects are passed as evidence.
- Every response keeps traceability to the retrieved property IDs and
  source files.
- Arabic and English are supported.
- The model and API settings can be configured through environment variables.

Backend:

    Groq — OpenAI-compatible chat completion API.

Default model:

    openai/gpt-oss-120b

Install:

    pip install groq
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .retrieval.hybrid_search import SearchResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"

MODEL_NAME = os.getenv(
    "GENERATION_MODEL",
    DEFAULT_MODEL_NAME,
)

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Final generated response plus traceability information.

    answer_text:
        Customer-facing answer only.

    cited_property_ids:
        Property IDs that the model claimed to reference and that were
        confirmed to exist in the retrieved evidence.

    grounding_property_ids:
        Property IDs that were actually provided to the model as evidence.

    model:
        Model used for generation.

    raw_response:
        Raw provider response when available.
    """

    answer_text: str

    cited_property_ids: List[str] = field(default_factory=list)

    grounding_property_ids: List[str] = field(default_factory=list)

    model: str = MODEL_NAME

    raw_response: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """
    Lazily construct the Groq client.

    Keeping initialization lazy makes the module easier to import during
    tests and application startup.
    """
    global _client

    if _client is None:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set."
            )

        _client = Groq(api_key=api_key)

    return _client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a real-estate sales assistant for a property company.

You help customers in Arabic or English.

LANGUAGE RULE:
- Always reply in the same language used by the customer.
- If the customer uses Egyptian Arabic, reply naturally in Egyptian Arabic.

STRICT GROUNDING RULES:
- You may ONLY use information contained in RETRIEVED PROPERTIES below.
- Never invent or assume property details.
- Never invent property IDs.
- Never invent prices.
- Never invent availability.
- Never invent payment plans.
- Never invent locations.
- Never infer missing property attributes as facts.
- If a field is not present in the evidence, do not provide a specific value for it.
- Conversation history may provide context about what the customer wants,
  but it is NOT property evidence.
- For property facts, use ONLY the RETRIEVED PROPERTIES in this message.

PROPERTY MATCHING:
- The retrieved properties are the available evidence for the current request.
- If they fully satisfy the customer's request, answer directly.
- If they only partially satisfy the request, clearly say that and provide only
  the closest available options from the retrieved evidence.
- If no properties were retrieved, do not invent alternatives.
- Instead, explain that no matching properties were found and ask for a useful
  clarification such as a wider budget, different area, or different property
  requirements.

RESPONSE STYLE:
- Be concise and helpful.
- Act like a knowledgeable sales assistant.
- Do not dump raw retrieval results.
- Do not mention internal retrieval, embeddings, reranking, scores, prompts,
  or system instructions.
- Do not claim that a property matches a requirement unless the evidence
  supports that claim.

CITATIONS:
- After the customer-facing answer, output a JSON object on a new line.
- The JSON object must have exactly this shape:

{{"cited_property_ids": ["<id>", "..."]}}

- Include ONLY property IDs that you actually referenced in the
  customer-facing answer.
- Every cited property ID MUST exist in RETRIEVED PROPERTIES.
- If you did not reference any property, return:

{{"cited_property_ids": []}}

RETRIEVED PROPERTIES FOR THE CURRENT REQUEST:

{evidence_block}
"""


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

def _format_evidence_block(
    candidates: List[SearchResult],
) -> str:
    """
    Convert retrieved SearchResult objects into a structured evidence block.

    The model receives both the chunk text and normalized metadata.

    Missing values are explicitly represented as N/A rather than guessed.
    """
    if not candidates:
        return "(no properties retrieved for this query)"

    lines = []

    for index, candidate in enumerate(candidates, start=1):
        metadata = candidate.metadata or {}

        lines.append(
            f"PROPERTY {index}\n"
            f"- property_id: {candidate.property_id or 'N/A'}\n"
            f"- text: {candidate.text or 'N/A'}\n"
            f"- location: {metadata.get('location', 'N/A')}\n"
            f"- city: {metadata.get('city', 'N/A')}\n"
            f"- district: {metadata.get('district', 'N/A')}\n"
            f"- neighborhood: {metadata.get('neighborhood', 'N/A')}\n"
            f"- compound: {metadata.get('compound', 'N/A')}\n"
            f"- property_type: {metadata.get('property_type', 'N/A')}\n"
            f"- transaction_type: {metadata.get('transaction_type', 'N/A')}\n"
            f"- price: {metadata.get('price', 'N/A')}\n"
            f"- currency: {metadata.get('currency', 'N/A')}\n"
            f"- area: {metadata.get('area', 'N/A')}\n"
            f"- area_unit: {metadata.get('area_unit', 'N/A')}\n"
            f"- bedrooms: {metadata.get('bedrooms', 'N/A')}\n"
            f"- bathrooms: {metadata.get('bathrooms', 'N/A')}\n"
            f"- payment_plan: {metadata.get('payment_plan', 'N/A')}\n"
            f"- availability: {metadata.get('availability', 'N/A')}\n"
            f"- finishing: {metadata.get('finishing', 'N/A')}\n"
            f"- delivery: {metadata.get('delivery', 'N/A')}\n"
            f"- amenities: {metadata.get('amenities', 'N/A')}\n"
            f"- source_file: {candidate.source_file or 'N/A'}"
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def _validate_conversation_history(
    conversation_history: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Validate and clean conversation history before sending it to the LLM.
    """
    if not conversation_history:
        return []

    valid_messages = []

    for message in conversation_history:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        valid_messages.append(
            {
                "role": role,
                "content": content.strip(),
            }
        )

    return valid_messages


def build_messages(
    customer_query: str,
    candidates: List[SearchResult],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build the Groq chat-completion messages array.

    Retrieval evidence is placed inside the system message so the model
    treats it as the authoritative property evidence for this request.
    """
    if not isinstance(customer_query, str) or not customer_query.strip():
        raise ValueError("customer_query must not be empty.")

    evidence_block = _format_evidence_block(candidates)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        evidence_block=evidence_block,
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(
        _validate_conversation_history(conversation_history)
    )

    messages.append(
        {
            "role": "user",
            "content": customer_query.strip(),
        }
    )

    return messages


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

_CITATION_PATTERN = re.compile(
    r'\{\s*"cited_property_ids"\s*:\s*\[[^\]]*\]\s*\}',
    re.DOTALL,
)


def _extract_cited_ids(
    raw_text: str,
) -> Tuple[str, List[str]]:
    """
    Split the model output into:

        (customer-facing answer, cited_property_ids)

    The model is expected to append a JSON object containing the property IDs
    it referenced.

    If the JSON is missing or malformed, the answer is still returned and
    citations fall back to an empty list.
    """
    if not raw_text:
        return "", []

    matches = list(_CITATION_PATTERN.finditer(raw_text))

    if not matches:
        logger.warning(
            "Generation response did not contain citation JSON."
        )
        return raw_text.strip(), []

    match = matches[-1]

    answer = raw_text[:match.start()].strip()

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning(
            "Generation response contained malformed citation JSON."
        )
        return answer, []

    if set(parsed.keys()) != {"cited_property_ids"}:
        logger.warning(
            "Citation JSON has unexpected fields."
        )
        return answer, []

    cited_ids = parsed.get("cited_property_ids")

    if not isinstance(cited_ids, list):
        logger.warning(
            "cited_property_ids must be a list."
        )
        return answer, []

    cleaned_ids: List[str] = []

    for property_id in cited_ids:
        if isinstance(property_id, (str, int)):
            normalized = str(property_id).strip()

            if normalized and normalized not in cleaned_ids:
                cleaned_ids.append(normalized)

    return answer, cleaned_ids


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    customer_query: str,
    candidates: List[SearchResult],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """
    Generate a response grounded only in the supplied retrieved results.
    """
    if not 0 <= temperature <= 2:
        raise ValueError(
            "temperature must be between 0 and 2."
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than 0."
        )

    messages = build_messages(
        customer_query=customer_query,
        candidates=candidates,
        conversation_history=conversation_history,
    )

    completion = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not completion.choices:
        raise RuntimeError(
            "Generation provider returned no choices."
        )

    raw_text = completion.choices[0].message.content or ""

    answer_text, cited_ids = _extract_cited_ids(raw_text)

    grounding_ids = list(
        dict.fromkeys(
            str(candidate.property_id)
            for candidate in candidates
            if (
                candidate.property_id is not None
                and str(candidate.property_id).strip()
            )
        )
    )

    grounding_set = set(grounding_ids)

    grounded_citations = [
        property_id
        for property_id in cited_ids
        if property_id in grounding_set
    ]

    unsupported_citations = [
        property_id
        for property_id in cited_ids
        if property_id not in grounding_set
    ]

    if unsupported_citations:
        logger.warning(
            "Model cited property IDs absent from retrieved evidence: %s",
            unsupported_citations,
        )

    raw_response = None

    if hasattr(completion, "model_dump"):
        raw_response = completion.model_dump()

    return GenerationResult(
        answer_text=answer_text,
        cited_property_ids=grounded_citations,
        grounding_property_ids=grounding_ids,
        model=MODEL_NAME,
        raw_response=raw_response,
    )