# rag-chatbot/src/prompts/prompt_templates.py

from langchain_core.prompts import PromptTemplate

def create_rag_prompt_template():
    """Creates a prompt template for the RAG chain."""
    template = """
    You are an intelligent document assistant. Use the following pieces of retrieved context to answer the user's question.
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    Keep the answer concise and based on the provided context.

    Context: {context}

    Chat History: {chat_history}

    Question: {question}

    Helpful Answer:
    """
    return PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])