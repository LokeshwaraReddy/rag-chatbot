# rag-chatbot/src/ui/sidebar.py

import streamlit as st
from configs import config

def create_sidebar():
    """Creates the sidebar for the Streamlit UI."""
    with st.sidebar:
        st.title(config.UI_TITLE)
        #st.image(f"{config.ASSETS_DIR}/logo.png", use_column_width=True)
        
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or TXT files",
            type=config.SUPPORTED_DOC_TYPES,
            accept_multiple_files=True
        )

        st.header("Retrieval Settings")
        k = st.slider("Number of chunks to retrieve (K)", 1, 10, config.RETRIEVER_K, 1)

        st.header("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, config.LLM_MODEL_KWARGS['temperature'], 0.1)
        max_tokens = st.slider("Max New Tokens", 50, 2048, config.LLM_MODEL_KWARGS['max_new_tokens'], 50)
        
        return uploaded_files, k, temperature, max_tokens