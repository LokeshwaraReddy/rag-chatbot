# rag-chatbot/src/vectorstore/faiss_store.py

import os
from langchain_community.vectorstores import FAISS
from configs import config
from src.utils.logger import logger

class FaissStore:
    """Manages the FAISS vector store."""

    def __init__(self, embedding_function):
        """
        Initializes the FaissStore.

        Args:
            embedding_function: The function to generate embeddings.
        """
        self.index_path = os.path.join(config.VECTORSTORE_DIR, config.VECTORSTORE_INDEX_NAME)
        self.embedding_function = embedding_function
        self.db = None
        self._load_index()

    def _load_index(self):
        """Loads the FAISS index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                self.db = FAISS.load_local(self.index_path, self.embedding_function, allow_dangerous_deserialization=True)
                logger.info(f"FAISS index loaded from {self.index_path}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.db = None
        else:
            logger.info("No existing FAISS index found. A new one will be created.")

    def add_documents(self, documents: list):
        """
        Adds documents to the FAISS index.

        Args:
            documents (list): A list of document chunks.
        """
        try:
            if self.db:
                self.db.add_documents(documents)
                logger.info("Incrementally updated FAISS index.")
            else:
                self.db = FAISS.from_documents(documents, self.embedding_function)
                logger.info("Created a new FAISS index.")
            self._save_index()
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {e}")

    def _save_index(self):
        """Saves the FAISS index to disk."""
        if self.db:
            if not os.path.exists(config.VECTORSTORE_DIR):
                os.makedirs(config.VECTORSTORE_DIR)
            try:
                self.db.save_local(self.index_path)
                logger.info(f"FAISS index saved to {self.index_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")

    def get_retriever(self, search_type: str = config.RETRIEVER_SEARCH_TYPE, k: int = config.RETRIEVER_K):
        """
        Gets a retriever from the vector store.

        Args:
            search_type (str): The type of search to perform.
            k (int): The number of top documents to retrieve.

        Returns:
            A LangChain retriever instance.
        """
        if self.db:
            return self.db.as_retriever(search_type=search_type, search_kwargs={'k': k})
        return None