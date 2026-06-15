# rag-chatbot/src/embeddings/embedding_generator.py

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch
from configs import config
from src.utils.logger import logger

class EmbeddingGenerator:
    """Generates embeddings for text."""

    def __init__(self):
        """Initializes the EmbeddingGenerator."""
        self.model_name = config.EMBEDDING_MODEL_NAME
        self.model_kwargs = config.EMBEDDING_MODEL_KWARGS
        self.encode_kwargs = config.EMBEDDING_ENCODE_KWARGS
        self.device = self._get_device()
        self.model_kwargs['device'] = self.device
        
        try:
            self.embedding_model = HuggingFaceBgeEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs
            )
            logger.info(f"Embedding model '{self.model_name}' loaded on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _get_device(self) -> str:
        """Determines the device to use for embeddings (CUDA or CPU)."""
        if config.USE_CUDA and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_embedding_model(self):
        """Returns the loaded embedding model."""
        return self.embedding_model

    def get_gpu_utilization(self) -> str:
        """Gets GPU utilization if CUDA is available."""
        if self.device == "cuda":
            try:
                return f"{torch.cuda.get_device_name(0)} - Usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB / {torch.cuda.memory_reserved(0)/1024**2:.2f} MB"
            except Exception as e:
                logger.error(f"Could not get GPU utilization: {e}")
                return "N/A"
        return "Not using GPU"