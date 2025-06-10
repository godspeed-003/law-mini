import streamlit as st
import os
from dotenv import load_dotenv
from utils import DocumentProcessor
import tempfile

# --- Add this block to check Pydantic version ---
import pydantic
from packaging import version
if version.parse(pydantic.__version__).major >= 2:
    st.error(
        f"Pydantic v{pydantic.__version__} detected. "
        "Please downgrade to Pydantic v1.x for compatibility with LangChain and langchain_groq.\n\n"
        "Run in your terminal:\n"
        "  pip uninstall pydantic\n"
        "  pip install 'pydantic<2.0.0'"
    )
    st.stop()
# --- End of block ---

# Load environment variables
load_dotenv()

# Get API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]

# For Google Cloud Vision credentials
import json
try:
    gcp_creds = st.secrets["gcp_service_account"]
    # Convert AttrDict to regular dict for JSON serialization
    gcp_creds_dict = {key: value for key, value in gcp_creds.items()}
    # Create temporary credentials file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(gcp_creds_dict, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
except Exception as e:
    st.error(f"Error setting up Google Cloud credentials: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Document Chat with Gemini",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
# Add chat history to session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Each item: {"role": "user"/"gemini", "content": str}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = True  # Flag to show chat interface

# Title and description
st.title("📚 Document Chat with Gemini")
st.markdown("""
This application allows you to upload multiple documents and chat with them using Google's Gemini AI.
Supported file types: PDF, DOCX, TXT, and CSV.
""")

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv'],
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Create temporary directory for uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    
                    # Show progress for file processing
                    progress_bar = st.progress(0)
                    for idx, uploaded_file in enumerate(uploaded_files):
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        file_paths.append(file_path)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Initialize document processor
                    processor = DocumentProcessor(api_key)
                    
                    # Process documents with status updates
                    st.text("Creating embeddings...")
                    vector_store = processor.process_documents(file_paths)
                    
                    st.text("Setting up QA chain...")
                    st.session_state.qa_chain = processor.create_qa_chain(vector_store)
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                    progress_bar.empty()
                    
            except ValueError as e:
                st.error(f"Processing error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error("Please check if your documents are valid and not empty.")
    
    st.markdown("---")
    st.header("Navigation")
    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.current_chat = True
        st.rerun()  # Changed from experimental_rerun

# Main content area - Full width chat layout
if st.session_state.qa_chain:
    st.markdown("""
        <style>
            .chat-outer {
                width: 100%;
                max-width: 900px;
                margin: 0 auto;
            }
            .chat-scrollbox {
                height: 500px;
                width: 100%;
                overflow-y: auto;
                border: 1px solid #eee;
                padding: 24px 18px 18px 18px;
                margin-bottom: 0;
                background-color: #f9f9f9;
                border-radius: 10px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03);
                display: flex;
                flex-direction: column;
                gap: 0.5em;
            }
            .user-message {
                background: #e6f0fa;
                padding: 14px 18px;
                border-radius: 10px;
                margin: 12px 0 8px 40px;
                color: #111;
                font-weight: 500;
                text-align: right;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            }
            .gemini-message {
                background: #fff;
                padding: 14px 18px;
                border-radius: 10px;
                margin: 8px 40px 12px 0;
                color: #222;
                font-weight: 400;
                border-left: 4px solid #4f8cff;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            }
            .expert-message {
                background: #f5f7fa;
                padding: 14px 18px;
                border-radius: 10px;
                margin: 8px 40px 12px 0;
                color: #1a237e;
                font-weight: 500;
                border-left: 4px solid #43a047;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            }
            .chat-input-area input {
                background: #fff !important;
                color: #111 !important;
                border-radius: 8px !important;
                border: 1px solid #ddd !important;
                font-size: 1.1em !important;
            }
            .chat-input-area label {
                color: #111 !important;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-outer">', unsafe_allow_html=True)
    st.subheader("💬 Chat")
    # Only show chat history container if there are messages
    if st.session_state.chat_history:
        st.markdown('<div class="chat-scrollbox">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-message'>You:<br>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "expert":
                st.markdown(
                    f"<div class='expert-message'>Legal Expert:<br>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='gemini-message'>Gemini:<br>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Input area at the bottom
    with st.form(key="chat_form"):
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        user_question = st.text_input("Ask a question:", key="chat_input", placeholder="Type your message here...")
        st.markdown('</div>', unsafe_allow_html=True)
        send, clear = st.columns([1, 1])
        send_clicked = send.form_submit_button("Send")
        clear_clicked = clear.form_submit_button("Clear Chat")
        if send_clicked and user_question.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.spinner("Gemini is thinking..."):
                try:
                    response = st.session_state.qa_chain(user_question)
                    # Only show expert output if it's not the "not available" message
                    if "expert" in response and response["expert"] and "not available" not in response["expert"]:
                        st.session_state.chat_history.append({
                            "role": "expert",
                            "content": response["expert"]
                        })
                    st.session_state.chat_history.append({
                        "role": "gemini",
                        "content": response["result"]
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        if clear_clicked:
            st.session_state.chat_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Please upload and process some documents to start chatting!")

# Footer
st.markdown("---")
st.markdown("Made with Streamlit and Google Gemini AI")