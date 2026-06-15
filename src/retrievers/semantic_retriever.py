# rag-chatbot/src/retrievers/semantic_retriever.py

from src.vectorstore.faiss_store import FaissStore
from configs import config

class SemanticRetriever:
    """Handles semantic retrieval from the vector store."""

    def __init__(self, vector_store: FaissStore):
        """
        Initializes the SemanticRetriever.

        Args:
            vector_store (FaissStore): An instance of the FaissStore.
        """
        self.vector_store = vector_store

    def get_retriever(self, search_type: str = config.RETRIEVER_SEARCH_TYPE, k: int = config.RETRIEVER_K):
        """
        Gets a retriever with specified parameters.

        Args:
            search_type (str): The type of search (e.g., 'similarity').
            k (int): The number of documents to retrieve.

        Returns:
            A retriever instance.
        """
        return self.vector_store.get_retriever(search_type=search_type, k=k)