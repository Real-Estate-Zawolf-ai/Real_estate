from app.agents.state import ConversationState

class ConversationalState(
    ConversationState,
    total=False,
):
    conversation_context: str
