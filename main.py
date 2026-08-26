import streamlit as st
import os
import rag_pipeline

st.set_page_config(page_title="Document QA Assistant", page_icon="📚", layout="wide")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📂 Document Manager")
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(rag_pipeline.DATA_DIR, uploaded_file.name)
            # Only process if new
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    rag_pipeline.process_and_store_pdf(save_path)
                st.success(f"Added: {uploaded_file.name}")
                st.rerun()

    st.divider()
    st.subheader("📑 Active Documents")
    
    current_files = rag_pipeline.get_all_uploaded_files()
    if current_files:
        for file_name in current_files:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {file_name}")
            if col2.button("🗑️", key=f"del_{file_name}", help=f"Delete {file_name}"):
                rag_pipeline.delete_file_and_embeddings(file_name)
                st.toast(f"Deleted {file_name}")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

# --- Main App: Chat Interface ---
st.title("🤖 Chat with Your Documents")
st.caption("Ask questions across all uploaded PDFs with source citations.")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User question input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = rag_pipeline.search_and_generate(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})