HEAD
# Intelligent Document Assistant using RAG

An AI-powered chatbot that can answer questions from uploaded documents using Retrieval-Augmented Generation (RAG). This project is suitable for a university capstone, NVIDIA AI GPU internship demonstration, or a GitHub portfolio project.

## Project Objective

To build a robust and scalable AI assistant that allows users to chat with their documents (PDF, DOCX, TXT). The system uses state-of-the-art AI techniques to provide accurate and context-aware responses.

## Features

-   **Multi-Format Document Upload**: Supports PDF, DOCX, and TXT files.
-   **Advanced Document Processing**: Implements cleaning, chunking, and metadata extraction.
-   **High-Quality Embeddings**: Uses Sentence Transformers for generating dense vector embeddings.
-   **Efficient Vector Storage**: Leverages FAISS for fast similarity searches.
-   **Configurable Retriever**: Allows tuning of retrieval parameters like top-K.
-   **Powerful LLM Integration**: Supports Llama 3 and Mistral Instruct models.
-   **Conversation Memory**: Maintains chat history for contextual conversations.
-   **Interactive UI**: A professional web interface built with Streamlit.
-   **Performance Monitoring**: Displays response time and source references.
-   **Optimized for NVIDIA GPUs**: Includes CUDA acceleration and GPU utilization monitoring.

## Architecture

The project follows a modular RAG architecture:

```mermaid
graph TD
    A[User] -- Uploads Document --> B(Streamlit UI);
    B -- Sends Document to Backend --> C(Document Loader);
    C -- Processes and Chunks Document --> D(Text Chunks);
    D -- Sends to Embedding Generator --> E(Embedding Generator);
    E -- Generates Embeddings --> F(Embeddings);
    F -- Stored in --> G(FAISS Vector Store);
    A -- Asks Question --> B;
    B -- Sends Question to Backend --> H(RAG Chain);
    H -- Retrieves Relevant Context --> G;
    H -- Generates Prompt --> I(LLM);
    I -- Generates Response --> H;
    H -- Sends Response to --> B;
    B -- Displays Response --> A;
# rag-chatbot
 4050712920d225b1e4b1c7fab93232f6c326b34b
