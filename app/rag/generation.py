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
- The LLM must never calculate, derive, estimate, or transform property facts.
- Financial values must only be shown when explicitly present in the evidence.
- Missing fields must remain missing; they must never be inferred.
- A post-generation safety layer removes unsupported financial claims.
- Normalized search requirements are provided to generation when available,
  so generation understands the criteria that produced the retrieved results.
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
            Property IDs explicitly cited by the model and verified against
            the retrieved evidence.

        grounding_property_ids:
            Property IDs provided to the model as evidence.

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
            raise RuntimeError("GROQ_API_KEY is not configured.")

        _client = Groq(api_key=api_key)

    return _client


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


def _get_model_name() -> str:
    """Return the configured generation model with a safe fallback."""
    model_name = settings.GENERATION_MODEL or settings.GROQ_MODEL

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

GROUNDING SOURCE:

The RETRIEVED PROPERTIES section is the only authoritative source for
property facts in this request.

NORMALIZED SEARCH REQUIREMENTS:

The NORMALIZED SEARCH REQUIREMENTS section contains the structured criteria
that were used by the application to retrieve and filter the properties.

Use these requirements to understand what the retrieval pipeline was asked
to search for.

IMPORTANT:

- Normalized search requirements describe search criteria, not additional
  property facts.
- Never use a requirement value as a property attribute unless that same
  value is explicitly present in the retrieved property evidence.
- If the customer's original wording differs from the normalized search
  requirements, do not silently pretend the two are identical.
- When useful, explain that the displayed properties match the normalized
  search criteria used by the system.
- Do not invent why a requirement was normalized or transformed.

STRICT GROUNDING RULES:

- Use ONLY information explicitly present in RETRIEVED PROPERTIES for
  property facts.
- Never invent property IDs.
- Never invent prices.
- Never invent availability.
- Never invent payment plans.
- Never invent locations.
- Never invent bedrooms, bathrooms, areas, developers, compounds, finishing,
  delivery dates, or amenities.
- Never infer a missing property attribute.
- Never calculate, derive, estimate, interpolate, or transform property facts.
- Never calculate down payments.
- Never calculate installments.
- Never calculate percentages, deposits, fees, discounts, or payment plans.
- Never convert currencies or units unless the exact converted value is
  explicitly present in the evidence.
- If a field is missing, empty, or N/A, do not present a specific value
  for that field.
- Numerical values must be copied exactly from the evidence.

FINANCIAL INFORMATION:

- A price may only be reported if explicitly present.
- A down payment may only be reported if explicitly present.
- A payment plan may only be reported if explicitly present.
- Installments may only be reported if explicitly present.
- Deposits may only be reported if explicitly present.
- Fees may only be reported if explicitly present.
- Discounts may only be reported if explicitly present.
- Never derive one financial value from another.
- Never calculate a financial value from a percentage.
- Never calculate a percentage from two financial values.

PROPERTY MATCHING:

- Retrieved properties are the available evidence for this response.
- Inspect the retrieved properties before deciding whether relevant results
  exist.
- If retrieved properties satisfy the customer's request based on explicit
  evidence, present those properties directly.
- If retrieved properties satisfy the NORMALIZED SEARCH REQUIREMENTS, they
  may be presented as matching the search criteria, but do not claim that
  they satisfy any additional criterion that is unsupported by evidence.
- If retrieved properties only partially satisfy the request, clearly explain
  the limitation and present only what the evidence supports.
- If no properties are retrieved, explain that no matching properties were
  found and ask for a useful clarification.
- Do not invent alternatives when no properties were retrieved.
- Do not silently replace the customer's requested location with another
  location.
- If the location in the customer's request differs from the location shown
  in retrieved properties, explicitly acknowledge the difference.
- Do not claim that a property satisfies a requirement unless the evidence
  explicitly supports that claim or the application explicitly provided the
  requirement as a normalized retrieval criterion.

IMPORTANT:

The application may retrieve properties using structured requirements that
are not word-for-word identical to the customer's original query.

When this happens:

1. Treat the normalized requirements as the criteria actually used for
   retrieval.
2. Treat RETRIEVED PROPERTIES as the only source of property facts.
3. Do not invent equivalence between different location names.
4. If the original query and retrieved property location visibly differ,
   explain that difference accurately.
5. Do not conclude that there are no results merely because the original
   wording differs from the normalized search criteria.

TABLE RULES:

- Tables are allowed when useful.
- Only include columns with explicitly available values.
- Every table value must come directly from the evidence.
- Never create calculated columns.
- Never create down-payment, installment, deposit, fee, discount, or payment
  plan columns unless the corresponding value is explicitly present in
  the evidence.

RESPONSE STYLE:

- Be concise and helpful.
- Act like a knowledgeable sales assistant.
- Do not dump raw retrieval results.
- Do not mention embeddings, reranking, prompts, system instructions,
  internal scores, or implementation details.
- Never claim unsupported facts.

CITATIONS:

At the very end of your response, output exactly one JSON object:

{{"cited_property_ids": ["<id>", "..."]}}

Rules for cited_property_ids:

- Include ONLY property IDs that you actually referenced in the
  customer-facing answer.
- Every cited ID MUST exist in RETRIEVED PROPERTIES.
- Never invent an ID.
- If no property was referenced, return:
  {{"cited_property_ids": []}}
- Do not put markdown fences around the JSON.
- Do not add any text after the JSON object.

NORMALIZED SEARCH REQUIREMENTS:

{requirements_block}

RETRIEVED PROPERTIES:

{evidence_block}
"""


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def _safe_value(value: Any) -> str:
    """
    Convert an evidence value to a deterministic display representation.

    No normalization or inference is performed.
    """
    if value is None:
        return "N/A"

    if isinstance(value, str):
        value = value.strip()
        return value if value else "N/A"

    return str(value)


def _format_requirements_block(
    requirements: dict[str, Any] | None,
) -> str:
    """
    Convert normalized search requirements into a deterministic context block.

    Requirements are presented as search criteria only. They are not treated
    as property evidence.
    """
    if not requirements:
        return "(no normalized search requirements were provided)"

    lines: list[str] = []

    for key, value in requirements.items():
        lines.append(
            f"- {key}: {_safe_value(value)}"
        )

    return "\n".join(lines)


def _format_evidence_block(
    candidates: list[SearchResult],
) -> str:
    """
    Convert retrieved SearchResult objects into a structured evidence block.

    Missing values are explicitly represented as N/A.
    """
    if not candidates:
        return "(no properties retrieved for this query)"

    lines: list[str] = []

    for index, candidate in enumerate(candidates, start=1):
        metadata = candidate.metadata or {}

        lines.append(
            f"PROPERTY {index}\n"
            f"- property_id: {_safe_value(candidate.property_id)}\n"
            f"- text: {_safe_value(candidate.text)}\n"
            f"- location: {_safe_value(metadata.get('location'))}\n"
            f"- city: {_safe_value(metadata.get('city'))}\n"
            f"- district: {_safe_value(metadata.get('district'))}\n"
            f"- neighborhood: {_safe_value(metadata.get('neighborhood'))}\n"
            f"- compound: {_safe_value(metadata.get('compound'))}\n"
            f"- developer: {_safe_value(metadata.get('developer'))}\n"
            f"- property_type: {_safe_value(metadata.get('property_type'))}\n"
            f"- transaction_type: {_safe_value(metadata.get('transaction_type'))}\n"
            f"- price: {_safe_value(metadata.get('price'))}\n"
            f"- currency: {_safe_value(metadata.get('currency'))}\n"
            f"- area: {_safe_value(metadata.get('area'))}\n"
            f"- area_unit: {_safe_value(metadata.get('area_unit'))}\n"
            f"- bedrooms: {_safe_value(metadata.get('bedrooms'))}\n"
            f"- bathrooms: {_safe_value(metadata.get('bathrooms'))}\n"
            f"- payment_plan: {_safe_value(metadata.get('payment_plan'))}\n"
            f"- availability: {_safe_value(metadata.get('availability'))}\n"
            f"- finishing: {_safe_value(metadata.get('finishing'))}\n"
            f"- delivery: {_safe_value(metadata.get('delivery'))}\n"
            f"- amenities: {_safe_value(metadata.get('amenities'))}\n"
            f"- source_file: {_safe_value(candidate.source_file)}"
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

    Only user and assistant messages are accepted.

    System messages are excluded because this module owns the system prompt.
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
    requirements: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Build the Groq chat-completion messages array.

    Retrieved evidence is placed inside the system message so the model
    treats it as authoritative property evidence for this request.

    Normalized requirements are also included as search context when
    available, but they are explicitly distinguished from property evidence.
    """
    if not isinstance(customer_query, str) or not customer_query.strip():
        raise ValueError("customer_query must not be empty.")

    evidence_block = _format_evidence_block(candidates)
    requirements_block = _format_requirements_block(requirements)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        requirements_block=requirements_block,
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

    The citation JSON must be present at the end of the model response.
    """
    if not raw_text:
        return "", []

    matches = list(_CITATION_PATTERN.finditer(raw_text))

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
# Citation validation
# ---------------------------------------------------------------------------


def _validate_citations(
    cited_ids: list[str],
    candidates: list[SearchResult],
) -> list[str]:
    """
    Keep only citation IDs that exist in the retrieved evidence.

    This is a hard safety boundary between model output and traceability.
    """
    grounding_ids = list(
        dict.fromkeys(
            str(candidate.property_id).strip()
            for candidate in candidates
            if (
                candidate.property_id is not None
                and str(candidate.property_id).strip()
            )
        )
    )

    grounding_set = set(grounding_ids)

    valid_ids: list[str] = []
    unsupported_ids: list[str] = []

    for property_id in cited_ids:
        normalized_id = str(property_id).strip()

        if normalized_id in grounding_set:
            if normalized_id not in valid_ids:
                valid_ids.append(normalized_id)
        else:
            unsupported_ids.append(normalized_id)

    if unsupported_ids:
        logger.warning(
            "Model cited property IDs absent from retrieved evidence: %s",
            unsupported_ids,
        )

    return valid_ids


# ---------------------------------------------------------------------------
# Financial evidence validation
# ---------------------------------------------------------------------------


_FINANCIAL_FIELDS = (
    "down_payment",
    "payment_plan",
    "installments",
    "deposit",
    "fees",
    "discount",
)


def _has_explicit_field(
    candidates: list[SearchResult],
    field_name: str,
) -> bool:
    """
    Return True only when at least one retrieved property explicitly contains
    a non-empty value for the requested field.

    No value is inferred from another field.
    """
    for candidate in candidates:
        metadata = candidate.metadata or {}
        value = metadata.get(field_name)

        if value is None:
            continue

        if isinstance(value, str):
            if value.strip() and value.strip().upper() != "N/A":
                return True
        else:
            return True

    return False


def _remove_markdown_table_column(
    text: str,
    header_pattern: re.Pattern[str],
) -> str:
    """
    Remove a markdown table column when the corresponding evidence field
    does not exist.

    This is intentionally conservative and only operates on markdown tables.
    """
    lines = text.splitlines()

    table_indexes = [
        index
        for index, line in enumerate(lines)
        if "|" in line
    ]

    if len(table_indexes) < 2:
        return text

    for index in table_indexes:
        line = lines[index]

        if not header_pattern.search(line):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        column_index = None

        for cell_index, cell in enumerate(cells):
            if header_pattern.search(cell):
                column_index = cell_index
                break

        if column_index is None:
            continue

        start = index
        end = index

        while start > 0 and "|" in lines[start - 1]:
            start -= 1

        while end + 1 < len(lines) and "|" in lines[end + 1]:
            end += 1

        for row_index in range(start, end + 1):
            row = lines[row_index]

            if "|" not in row:
                continue

            row_cells = [
                cell.strip()
                for cell in row.strip().strip("|").split("|")
            ]

            if column_index >= len(row_cells):
                continue

            del row_cells[column_index]

            lines[row_index] = "| " + " | ".join(row_cells) + " |"

    return "\n".join(lines)


def _remove_unsupported_financial_claims(
    answer_text: str,
    candidates: list[SearchResult],
) -> str:
    """
    Remove financial claims that are not explicitly supported by evidence.

    This is a post-generation safety layer. It exists because an LLM can
    occasionally violate a prompt rule even when the rule is explicit.

    The function is conservative:

    - It only removes known unsupported financial claims.
    - It does not modify supported price/area/bedroom/location values.
    - It removes unsupported financial table columns when the field is absent
      from all retrieved candidates.
    """
    if not answer_text:
        return answer_text

    sanitized = answer_text

    field_patterns: dict[str, re.Pattern[str]] = {
        "down_payment": re.compile(
            r"(down[\s-]?payment|down payment|مقدم|دفعة\s+مقدمة)",
            re.IGNORECASE,
        ),
        "payment_plan": re.compile(
            r"(payment\s+plan|payment\s+plans|خطة\s+السداد|نظام\s+السداد)",
            re.IGNORECASE,
        ),
        "installments": re.compile(
            r"(installment|installments|قسط|أقساط)",
            re.IGNORECASE,
        ),
        "deposit": re.compile(
            r"(deposit|حجز|وديعة)",
            re.IGNORECASE,
        ),
        "fees": re.compile(
            r"(fees?|رسوم|مصاريف)",
            re.IGNORECASE,
        ),
        "discount": re.compile(
            r"(discount|discounts|خصم|خصومات)",
            re.IGNORECASE,
        ),
    }

    for field_name, pattern in field_patterns.items():
        if _has_explicit_field(candidates, field_name):
            continue

        # Remove unsupported financial columns from markdown tables.
        sanitized = _remove_markdown_table_column(
            sanitized,
            pattern,
        )

        # Remove standalone lines that are clearly dedicated to the
        # unsupported financial field.
        lines = sanitized.splitlines()
        filtered_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                filtered_lines.append(line)
                continue

            # Markdown table rows are handled separately above.
            if "|" in stripped:
                filtered_lines.append(line)
                continue

            # Remove simple "Down payment: 123,456" style claims.
            if pattern.search(stripped):
                if re.search(
                    r"[:：]\s*"
                    r"(?:(?:EGP|USD|EUR|AED|\$|€|£)\s*)?"
                    r"\d[\d,.\s]*",
                    stripped,
                    re.IGNORECASE,
                ):
                    logger.warning(
                        "Removed unsupported financial claim for field '%s'.",
                        field_name,
                    )
                    continue

                # Also remove short standalone financial statements.
                if len(stripped.split()) <= 12:
                    logger.warning(
                        "Removed unsupported financial statement for field '%s'.",
                        field_name,
                    )
                    continue

            filtered_lines.append(line)

        sanitized = "\n".join(filtered_lines)

    sanitized = re.sub(
        r"\n{3,}",
        "\n\n",
        sanitized,
    )

    return sanitized.strip()


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
        raise ValueError("temperature must be between 0 and 2.")

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
    requirements: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """
    Generate a response grounded only in the supplied retrieved results.

    Args:
        customer_query:
            Original customer request.

        candidates:
            Final retrieved/reranked property evidence.

        requirements:
            Structured/normalized search requirements used by the retrieval
            pipeline. Optional for backward compatibility with existing
            callers and tests.

        conversation_history:
            Optional previous user/assistant messages.

        temperature:
            Provider generation temperature.

        max_tokens:
            Maximum generated tokens.

    This function does not perform retrieval or access PostgreSQL.
    """
    _validate_generation_parameters(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    messages = build_messages(
        customer_query=customer_query,
        candidates=candidates,
        requirements=requirements,
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

    answer_text, cited_ids = _extract_cited_ids(raw_text)

    # -----------------------------------------------------------------------
    # Hard post-generation grounding guardrail.
    #
    # The prompt tells the LLM not to invent financial values, but the
    # application must not rely on prompt compliance alone.
    # -----------------------------------------------------------------------

    sanitized_answer = _remove_unsupported_financial_claims(
        answer_text=answer_text,
        candidates=candidates,
    )

    if sanitized_answer != answer_text:
        logger.warning(
            "generation.py: removed unsupported financial claims "
            "from model output."
        )

    grounding_ids = list(
        dict.fromkeys(
            str(candidate.property_id).strip()
            for candidate in candidates
            if (
                candidate.property_id is not None
                and str(candidate.property_id).strip()
            )
        )
    )

    grounded_citations = _validate_citations(
        cited_ids=cited_ids,
        candidates=candidates,
    )

    raw_response: dict[str, Any] | None = None

    if hasattr(completion, "model_dump"):
        raw_response = completion.model_dump()

    return GenerationResult(
        answer_text=sanitized_answer,
        cited_property_ids=grounded_citations,
        grounding_property_ids=grounding_ids,
        model=model_name,
        raw_response=raw_response,
    )