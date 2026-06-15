# rag-chatbot/src/memory/conversation_memory.py

from langchain.memory import ConversationBufferWindowMemory
from configs import config

def get_conversation_memory():
    """Creates and returns a conversation memory buffer."""
    return ConversationBufferWindowMemory(
        k=config.MEMORY_WINDOW_SIZE,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )