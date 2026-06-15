# rag-chatbot/src/ui/chat_interface.py

import streamlit as st
import time

def display_chat_history():
    """Displays the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(rag_chain):
    """Handles user input and generates a response."""
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            start_time = time.time()
            
            with st.spinner("Thinking..."):
                response = rag_chain({"question": prompt})
            
            end_time = time.time()
            response_time = end_time - start_time
            
            full_response += response["answer"]
            message_placeholder.markdown(full_response + "▌")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            with st.expander("Sources"):
                for source in response["source_documents"]:
                    st.write(f"- {source.metadata['source']}:")
                    st.info(source.page_content)
            
            st.info(f"Response time: {response_time:.2f} seconds")
            st.info(f"Retrieved chunks: {len(response['source_documents'])}")