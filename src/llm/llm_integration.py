# rag-chatbot/src/llm/llm_integration.py

from langchain_community.llms import Ollama
from configs import config
from src.utils.logger import logger

class LLMIntegration:
    """Handles LLM integration."""

    def __init__(self):
        """Initializes the LLMIntegration."""
       
        
        try:
            self.llm = Ollama(
                model="llama3.2"
            )
            logger.info(f"LLM '{config.LLM_REPO_ID}' initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def get_llm(self):
        """Returns the initialized LLM."""
        return self.llm