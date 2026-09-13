SYSTEM_PROMPT = """
You are a helpful and reliable AI assistant.

Follow these rules:
- Give accurate answers.
- Do not invent information.
- If you do not know something, say so.
- Follow the instructions provided in the user prompt.
"""


def build_question_prompt(question: str) -> str:
    return f"""
Answer the following question:

{question}
"""


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    return f"""
Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

Rules:
- Use only information from the context.
- Do not invent information.
- If the answer cannot be found in the context, say that
  there is not enough information in the provided context.
"""
