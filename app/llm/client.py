import os
from typing import Type, TypeVar
from app.llm.prompts import build_rag_prompt,build_question_prompt,SYSTEM_PROMPT
from app.llm.structured_output import AnswerOutput, ClassificationOutput
from groq import Groq
from pydantic import BaseModel
from app.helpers.config import get_settings


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        api_key: get_settings().GROQ_API_KEY | None = None,
        model: get_settings().GENERATION_MODEL_ID | None = None,
    ):
        self.client = Groq(
            api_key=api_key or os.getenv("GROQ_API_KEY")
        )

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

    def generate(
        self,
        prompt: build_rag_prompt | build_question_prompt,
        system_prompt: SYSTEM_PROMPT,
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
        output_schema: AnswerOutput | ClassificationOutput,
        system_prompt: SYSTEM_PROMPT,
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