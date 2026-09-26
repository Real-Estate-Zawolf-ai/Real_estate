from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    answer: str = Field(
        description="The final answer to the user's question."
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score between 0 and 1."
    )


class ClassificationOutput(BaseModel):
    category: str = Field(
        description="The classification category."
    )

    reasoning: str = Field(
        description="Brief explanation for the classification."
    )
