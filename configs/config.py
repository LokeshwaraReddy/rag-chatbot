# rag-chatbot/configs/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- General ---
APP_NAME = "Intelligent Document Assistant"
APP_VERSION = "1.0.0"

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADED_FILES_DIR = os.path.join(DATA_DIR, "uploaded_files")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Document Processing ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_DOC_TYPES = ["pdf", "docx", "txt"]

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_KWARGS = {'device': 'cuda'}
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}

# --- Vector Database ---
VECTORSTORE_INDEX_NAME = "document_index"

# --- Retriever ---
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K = 5

# --- LLM ---
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # or "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MODEL_KWARGS = {
    "temperature": 0.5,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1
}
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Conversation Memory ---
MEMORY_WINDOW_SIZE = 5

# --- UI ---
UI_TITLE = "Intelligent Document Assistant"
UI_LAYOUT = "wide"
UI_INITIAL_SIDEBAR_STATE = "expanded"

# --- Logging ---
LOG_FILE_PATH = os.path.join(LOGS_DIR, "app.log")
LOG_LEVEL = "INFO"

# --- GPU ---
USE_CUDA = True