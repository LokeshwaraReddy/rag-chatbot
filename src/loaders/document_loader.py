# rag-chatbot/src/loaders/document_loader.py

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from configs import config
from src.utils.logger import logger

class DocumentLoader:
    """Handles loading and processing of documents."""

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        """
        Initializes the DocumentLoader.

        Args:
            chunk_size (int): The size of text chunks.
            chunk_overlap (int): The overlap between text chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def load_and_split_documents(self, file_paths: List[str]) -> List:
        """
        Loads documents from file paths and splits them into chunks.

        Args:
            file_paths (List[str]): A list of paths to the documents.

        Returns:
            List: A list of document chunks.
        """
        documents = []
        for file_path in file_paths:
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file_path.endswith(".txt"):
                    loader = TextLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}")
                continue
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Loaded and split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks