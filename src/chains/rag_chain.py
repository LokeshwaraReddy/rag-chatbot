# rag-chatbot/src/chains/rag_chain.py

from langchain.chains import create_history_aware_retriever,create_retrieval_chain,ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.llm.llm_integration import LLMIntegration
from src.prompts.prompt_templates import create_rag_prompt_template
from src.memory.conversation_memory import get_conversation_memory
from src.utils.logger import logger

class RAGChain:
    """Constructs and manages the RAG chain."""

    def __init__(self, retriever):
        """
        Initializes the RAGChain.

        Args:
            retriever: The document retriever.
        """
        self.retriever = retriever
        self.llm = LLMIntegration().get_llm()
        self.prompt = create_rag_prompt_template()
        self.memory = get_conversation_memory()

    def create_chain(self):
        """
        Creates the ConversationalRetrievalChain.

        Returns:
            An instance of ConversationalRetrievalChain.
        """
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self.prompt},
                return_source_documents=True
            )
            logger.info("ConversationalRetrievalChain created successfully.")
            return chain
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            return None