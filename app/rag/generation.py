"""
generation.py

Final step of the RAG pipeline: takes the customer's query + the
reranked, retrieved property evidence and generates a grounded
response.

Per the project's AI design principles:
    - The LLM must NOT invent property IDs, prices, availability, or
      payment plans — it may only use what's in the retrieved evidence.
    - Retrieval and generation stay separate (this module never
      searches anything itself — it only consumes SearchResult objects
      handed to it by retrieval/hybrid_search.py + reranker.py).
    - Every generated answer keeps a traceable link back to the
      property_ids/source_files it was grounded in.

Backend: Groq (OpenAI-compatible API), model "openai/gpt-oss-120b".

"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..rag.retrieval.hybrid_search import SearchResult  

MODEL_NAME = "openai/gpt-oss-120b"

# Keep generation deterministic-leaning: this is a business tool, not a
# creative one. Low temperature reduces (never eliminates) the risk of
# the model drifting from the grounded evidence.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800


@dataclass
class GenerationResult:
    """The final answer plus everything needed to trace and audit it."""
    answer_text: str
    cited_property_ids: List[str] = field(default_factory=list)
    grounding_property_ids: List[str] = field(default_factory=list)  # what was actually offered as evidence
    model: str = MODEL_NAME
    raw_response: Optional[dict] = None


_client = None


def _get_client():
    """Lazily construct the Groq client so importing this module never
    requires GROQ_API_KEY to be set (e.g. during unit tests that mock
    the client entirely).
    """
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


SYSTEM_PROMPT_TEMPLATE = """You are a real-estate sales assistant for a property company.
You help customers in Arabic or English — always reply in the same language the customer used.

STRICT GROUNDING RULES:
- You may ONLY mention properties, prices, availability, payment plans, and locations that appear
  in the "RETRIEVED PROPERTIES" data below. Never invent or assume any property detail.
- If the retrieved properties don't fully satisfy the customer's request, say so honestly and
  suggest the closest available options from the retrieved data, or ask a clarifying question.
- If no properties were retrieved, do not invent any — tell the customer no matching properties
  were found and ask a clarifying question (e.g. a wider budget or different area).
- Never state a specific price, unit number, or availability status unless it is explicitly present
  in the retrieved data.
- Keep responses concise and helpful, like a knowledgeable sales assistant — not a search engine
  dumping results.

RETRIEVED PROPERTIES:
{evidence_block}

After your reply to the customer, on a new line, output ONLY a JSON object of this exact shape
(no markdown fences, no extra text) listing the property_ids you actually referenced in your reply:
{{"cited_property_ids": ["<id>", "..."]}}
"""


def _format_evidence_block(candidates: List[SearchResult]) -> str:
    if not candidates:
        return "(no properties retrieved for this query)"

    lines = []
    for c in candidates:
        meta = c.metadata or {}
        lines.append(
            f"- property_id: {c.property_id}\n"
            f"  text: {c.text}\n"
            f"  location: {meta.get('location', 'N/A')}\n"
            f"  price: {meta.get('price', 'N/A')}\n"
            f"  bedrooms: {meta.get('bedrooms', 'N/A')}\n"
            f"  payment_plan: {meta.get('payment_plan', 'N/A')}\n"
            f"  availability: {meta.get('availability', 'N/A')}\n"
            f"  source_file: {c.source_file or 'N/A'}"
        )
    return "\n".join(lines)


def build_messages(
    customer_query: str,
    candidates: List[SearchResult],
    conversation_history: Optional[List[dict]] = None,
) -> List[dict]:
    """Build the Groq chat-completion `messages` array.

    `conversation_history` (optional) is a list of prior turns as
    {"role": "user"|"assistant", "content": str}, so multi-turn context
    (customer already said their budget last message, etc.) is preserved.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        evidence_block=_format_evidence_block(candidates)
    )
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": customer_query})
    return messages


def _extract_cited_ids(raw_text: str) -> tuple[str, List[str]]:
    """Split the model's reply into (customer-facing answer, cited_ids),
    parsing the trailing JSON block requested in the system prompt.
    Falls back gracefully — a malformed/missing JSON block yields an
    empty cited-ids list rather than crashing the whole request.
    """
    match = re.search(r'\{"cited_property_ids"\s*:\s*\[.*?\]\}', raw_text, re.DOTALL)
    if not match:
        return raw_text.strip(), []

    answer = raw_text[: match.start()].strip()
    try:
        parsed = json.loads(match.group(0))
        cited = parsed.get("cited_property_ids", [])
    except json.JSONDecodeError:
        cited = []
    return answer, cited


def generate_response(
    customer_query: str,
    candidates: List[SearchResult],
    conversation_history: Optional[List[dict]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """Generate the final grounded reply to the customer.

    `candidates` should be the output of retrieval/reranker.rerank() —
    already filtered, retrieved, and reranked. This function does not
    retrieve anything itself.
    """
    messages = build_messages(customer_query, candidates, conversation_history)

    client = _get_client()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_text = completion.choices[0].message.content
    answer_text, cited_ids = _extract_cited_ids(raw_text)

    # Traceability guard: warn (don't silently trust) if the model cited
    # a property_id that wasn't actually in the retrieved evidence —
    # a sign of hallucination that should be logged/monitored per the
    # project's observability requirements.
    grounding_ids = [c.property_id for c in candidates]
    hallucinated = [pid for pid in cited_ids if pid not in grounding_ids]
    if hallucinated:
        import logging
        logging.getLogger(__name__).warning(
            "generation.py: model cited property_ids not present in retrieved "
            "evidence (possible hallucination): %s", hallucinated,
        )

    return GenerationResult(
        answer_text=answer_text,
        cited_property_ids=cited_ids,
        grounding_property_ids=grounding_ids,
        model=MODEL_NAME,
        raw_response=completion.model_dump() if hasattr(completion, "model_dump") else None,
    )