# rag-chatbot/app.py

import streamlit as st
import os
import time
from configs import config
from src.loaders.document_loader import DocumentLoader
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.faiss_store import FaissStore
from src.chains.rag_chain import RAGChain
from src.ui.sidebar import create_sidebar
from src.ui.chat_interface import display_chat_history, handle_user_input
from src.utils.logger import logger

def initialize_system():
    """Initializes all the necessary components for the RAG system."""
    if "embedding_generator" not in st.session_state:
        st.session_state.embedding_generator = EmbeddingGenerator()
    
    if "vector_store" not in st.session_state:
        embedding_model = st.session_state.embedding_generator.get_embedding_model()
        st.session_state.vector_store = FaissStore(embedding_model)
    
    if "document_loader" not in st.session_state:
        st.session_state.document_loader = DocumentLoader()

def process_uploaded_files(uploaded_files):
    """Processes uploaded files, creates embeddings, and updates the vector store."""
    if uploaded_files:
        with st.spinner("Processing documents..."):
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(config.UPLOADED_FILES_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            chunks = st.session_state.document_loader.load_and_split_documents(file_paths)
            st.session_state.vector_store.add_documents(chunks)
            st.success("Documents processed and indexed successfully!")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=config.UI_TITLE,
        layout=config.UI_LAYOUT,
        initial_sidebar_state=config.UI_INITIAL_SIDEBAR_STATE
    )

    if not os.path.exists(config.UPLOADED_FILES_DIR):
        os.makedirs(config.UPLOADED_FILES_DIR)
        
    initialize_system()
    
    uploaded_files, k, temperature, max_tokens = create_sidebar()
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)

    retriever = st.session_state.vector_store.get_retriever(k=k)
    
    if retriever:
        rag_chain_instance = RAGChain(retriever)
        # Update LLM params
        rag_chain_instance.llm.temperature=temperature
        rag_chain_instance.llm.num_predict= max_tokens
        
        rag_chain = rag_chain_instance.create_chain()
        
        st.title("Chat with your Documents")
        display_chat_history()
        handle_user_input(rag_chain)
    else:
        st.info("Please upload documents to begin chatting.")

    with st.sidebar:
        st.header("System Status")
        st.info(f"GPU Utilization: {st.session_state.embedding_generator.get_gpu_utilization()}")


if __name__ == "__main__":
    main()