from typing_extensions import TypedDict
from app.graph.state import ConversationState

class PropertyState(
    ConversationState,
    total=False,
):
    conversation_context: str