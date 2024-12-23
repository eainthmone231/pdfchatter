import os
import uuid
import json
import shutil
from pathlib import Path
import markdown
import streamlit as st
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from byaldi import RAGMultiModalModel
from logger import get_logger

# Set up directories
UPLOAD_FOLDER = Path("uploaded_documents")
INDEX_FOLDER = Path(".byaldi")
SESSION_FOLDER = Path("sessions")

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
INDEX_FOLDER.mkdir(parents=True, exist_ok=True)
SESSION_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize logger
logger = get_logger(__name__)
st.title("PDF Chatterbot")
st.markdown(
    """
   PDF Chatter Bot: An AI-powered chatbot designed to make your documents more interactive and accessible. 
   Simply upload your PDF documents, and the bot will help you explore and retrieve answers from your content with ease.
    """
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "RAG_models" not in st.session_state:
    st.session_state.RAG_models = {}

# Utility functions
def load_rag_model_for_session(session_id):
    index_path = INDEX_FOLDER / session_id
    if index_path.exists():
        try:
            RAG = RAGMultiModalModel.from_index(str(index_path))
            st.session_state.RAG_models[session_id] = RAG
            logger.info(f"Loaded RAG model for session {session_id}.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")

# Load existing indexes on app start
for session_dir in INDEX_FOLDER.iterdir():
    if session_dir.is_dir():
        load_rag_model_for_session(session_dir.name)

# UI Components
st.sidebar.header("Session Management")
session_id = st.session_state.session_id
session_file = SESSION_FOLDER / f"{session_id}.json"

# Load chat history
if session_file.exists():
    with open(session_file, "r") as f:
        session_data = json.load(f)
    chat_history = session_data.get("chat_history", [])
else:
    chat_history = []

# File Upload
uploaded_files = st.file_uploader(
    "Upload documents to index", accept_multiple_files=True
)
if st.button("Index Files"):
    if uploaded_files:
        session_folder = UPLOAD_FOLDER / session_id
        session_folder.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            file_path = session_folder / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        try:
            with st.spinner('Indexing files...'):
                index_name = session_id
                index_path = INDEX_FOLDER / index_name
                RAG = index_documents(str(session_folder), index_name=str(index_name), index_path=str(index_path))
                if RAG is None:
                    raise ValueError("Indexing failed.")
                st.session_state.RAG_models[session_id] = RAG
                st.success("Files indexed successfully!")
        except Exception as e:
            st.error(f"Error indexing documents: {e}")
    else:
        st.warning("No files uploaded.")

# Chat Interface
st.subheader("Chat")
user_query = st.text_input("Enter your query:")
if st.button("Send Query"):
    if user_query and session_id in st.session_state.RAG_models:
        try:
            with st.spinner('Generating response...'):
                RAG = st.session_state.RAG_models[session_id]
                retrieved_images = retrieve_documents(RAG, user_query, session_id)
                response_text, _ = generate_response(retrieved_images, user_query, session_id)
                chat_history.append({"role": "user", "content": user_query})
                chat_history.append({"role": "assistant", "content": response_text})
                st.success("Query processed successfully!")
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("No RAG model loaded for the session.")

# Display chat history
for message in chat_history:
    role = "User" if message["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {message['content']}")

# Save chat history
with open(session_file, "w") as f:
    json.dump({"chat_history": chat_history}, f)

# Session Reset
if st.sidebar.button("Reset Session"):
    if session_file.exists():
        session_file.unlink()
    session_folder = UPLOAD_FOLDER / session_id
    if session_folder.exists():
        shutil.rmtree(session_folder)
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()
