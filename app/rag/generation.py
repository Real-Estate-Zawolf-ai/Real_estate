"""
Grounded response generation for the AI Real Estate Assistant.

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
- Every response keeps traceability to retrieved property IDs and source files.
- Arabic and English are supported.
- Model and API settings are loaded from app.helpers.config.
- PostgreSQL is not accessed directly by this module.

Backend:

    Groq — OpenAI-compatible chat completion API.

Default model:

    openai/gpt-oss-120b
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..helpers.config import get_settings
from .retrieval.hybrid_search import SearchResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

settings = get_settings()

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """
    Final generated response plus traceability information.

    Attributes:
        answer_text:
            Customer-facing answer only.

        cited_property_ids:
            Property IDs that the model claimed to reference and that were
            verified against the retrieved evidence.

        grounding_property_ids:
            Property IDs that were actually provided to the model as evidence.

        model:
            Model used for generation.

        raw_response:
            Raw provider response when available.
    """

    answer_text: str
    cited_property_ids: list[str] = field(default_factory=list)
    grounding_property_ids: list[str] = field(default_factory=list)
    model: str = ""
    raw_response: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client: Any = None


def _get_client() -> Any:
    """
    Lazily construct the Groq client.

    Lazy initialization keeps the module easy to import during tests and
    application startup without requiring an API key immediately.
    """

    global _client

    if _client is None:
        from groq import Groq

        api_key = settings.GROQ_API_KEY

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        _client = Groq(api_key=api_key)

    return _client


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


def _get_model_name() -> str:
    """Return the configured generation model with a safe fallback."""

    model_name = (
        settings.GENERATION_MODEL
        or settings.GROQ_MODEL
    )

    if not model_name:
        raise RuntimeError(
            "No generation model is configured. "
            "Set GENERATION_MODEL or GROQ_MODEL."
        )

    return model_name


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

PROPERTY MATCHING:
- The retrieved properties are the available evidence.
- If they fully satisfy the customer's request, answer directly.
- If they only partially satisfy the customer's request, clearly say that and
  provide only the closest available options from the retrieved evidence.
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
- Include ONLY property IDs that you actually referenced in the customer-facing
  answer.
- Every cited property ID MUST exist in RETRIEVED PROPERTIES.
- If you did not reference any property, return:
{{"cited_property_ids": []}}

RETRIEVED PROPERTIES:

{evidence_block}
"""


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def _format_evidence_block(
    candidates: list[SearchResult],
) -> str:
    """
    Convert retrieved SearchResult objects into a structured evidence block.

    Missing values are explicitly represented as N/A rather than guessed.
    """

    if not candidates:
        return "(no properties retrieved for this query)"

    lines: list[str] = []

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
            f"- developer: {metadata.get('developer', 'N/A')}\n"
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
    conversation_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """
    Validate and clean conversation history before sending it to the LLM.

    Only user and assistant messages are accepted. System messages are
    intentionally excluded because this module owns the system prompt.
    """

    if not conversation_history:
        return []

    valid_messages: list[dict[str, str]] = []

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
    candidates: list[SearchResult],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
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

    messages: list[dict[str, str]] = [
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
) -> tuple[str, list[str]]:
    """
    Split model output into:

        (customer-facing answer, cited_property_ids)

    The citation block must contain the expected JSON object. If the block
    is missing or malformed, the answer is still returned and citations
    fall back to an empty list.
    """

    if not raw_text:
        return "", []

    matches = list(
        _CITATION_PATTERN.finditer(raw_text)
    )

    if not matches:
        return raw_text.strip(), []

    match = matches[-1]

    answer = raw_text[:match.start()].strip()

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning(
            "generation.py: failed to parse citation JSON."
        )
        return answer, []

    if not isinstance(parsed, dict):
        return answer, []

    if set(parsed.keys()) != {"cited_property_ids"}:
        logger.warning(
            "generation.py: citation JSON has an unexpected schema."
        )
        return answer, []

    cited_ids = parsed.get("cited_property_ids")

    if not isinstance(cited_ids, list):
        return answer, []

    normalized_ids: list[str] = []

    for property_id in cited_ids:
        if not isinstance(property_id, (str, int)):
            continue

        normalized_id = str(property_id).strip()

        if normalized_id:
            normalized_ids.append(normalized_id)

    return answer, list(dict.fromkeys(normalized_ids))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_generation_parameters(
    temperature: float,
    max_tokens: int,
) -> None:
    """Validate generation parameters before calling the provider."""

    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number.")

    if not 0 <= float(temperature) <= 2:
        raise ValueError(
            "temperature must be between 0 and 2."
        )

    if not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer.")

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than 0."
        )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_response(
    customer_query: str,
    candidates: list[SearchResult],
    conversation_history: list[dict[str, str]] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """
    Generate a response grounded only in the supplied retrieved results.

    This function does not perform retrieval or access PostgreSQL.
    """

    _validate_generation_parameters(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    messages = build_messages(
        customer_query=customer_query,
        candidates=candidates,
        conversation_history=conversation_history,
    )

    model_name = _get_model_name()

    completion = _get_client().chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=max_tokens,
    )

    if not completion.choices:
        raise RuntimeError(
            "Generation provider returned no completion choices."
        )

    raw_text = (
        completion.choices[0].message.content
        or ""
    )

    answer_text, cited_ids = _extract_cited_ids(
        raw_text
    )

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

    raw_response: dict[str, Any] | None = None

    if hasattr(completion, "model_dump"):
        raw_response = completion.model_dump()

    return GenerationResult(
        answer_text=answer_text,
        cited_property_ids=grounded_citations,
        grounding_property_ids=grounding_ids,
        model=model_name,
        raw_response=raw_response,
    )