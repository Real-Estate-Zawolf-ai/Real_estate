from typing import TypeVar
from app.llm.prompts import SYSTEM_PROMPT
from groq import Groq
from pydantic import BaseModel
from app.helpers.config import get_settings


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.client = Groq(api_key=api_key or settings.GROQ_API_KEY)
        self.model = model or settings.GENERATION_MODEL or settings.GROQ_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> T:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=output_schema,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise ValueError(
                "LLM returned no structured output."
            )

        return parsed