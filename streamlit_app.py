import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional
from langchain.schema.runnable import RunnablePassthrough

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="GPT-4o PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Add these constants near the top of the file
BASELINE_FILE_PATH = "data/baseline.pdf"  # or "data/baseline.txt"
BASELINE_NAME = "baseline.pdf"  # or "baseline.txt"

def initialize_baseline_db() -> Chroma:
    """Initialize vector DB with baseline knowledge file."""
    logger.info("Initializing baseline vector DB")
    try:
        loader = UnstructuredPDFLoader(BASELINE_FILE_PATH)  # or use TextLoader for txt files
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="myRAG"
        )
        logger.info("Baseline vector DB created")
        return vector_db
    except Exception as e:
        logger.error(f"Failed to initialize baseline DB: {e}")
        return None

def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Updated embeddings configuration
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="myRAG"
    )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_question(question: str, vector_db: Chroma) -> str:
    """
    Process a user question using the vector database and GPT-4o.
    """
    logger.info(f"Processing question: {question}")
    
    # Set up retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create the prompt
    prompt_text = f"""Answer the question based ONLY on the following context:
    
    Context:
    {context}
    
    Question: {question}
    """
    
    try:
        # Get completion from Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # Make sure this matches your deployment name
            messages=[
                {"role": "system", "content": "After analyzing the document based on the user's needs, it always provides output formatted for a .csv file. Never output results in a bullet point format. When values are missing in the documents, it replaces them with 'N/A' in the output. It always provides the information in a copyable .csv format only. Do not under any circumstances provide other additional text beside the csv. formatted text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        logger.info("Question processed and response generated")
        return answer
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        raise e


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """Delete the vector database and reinitialize with baseline if available."""
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("current_file", None)
        
        # Reinitialize with baseline
        st.session_state["vector_db"] = initialize_baseline_db()
        if st.session_state["vector_db"] is not None:
            st.success("Collection deleted and reset to baseline.")
        else:
            st.error("Failed to reinitialize baseline database.")
        
        logger.info("Vector DB reset to baseline")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def get_baseline_documents() -> List[str]:
    """Get a list of documents in the baseline knowledge base."""
    try:
        loader = UnstructuredPDFLoader(BASELINE_FILE_PATH)
        data = loader.load()
        return [BASELINE_NAME]  # Add more baseline documents if you have multiple
    except Exception as e:
        logger.error(f"Failed to get baseline documents: {e}")
        return []

def main() -> None:
    """Main function to run the Streamlit application."""
    st.subheader("📚 GPT-4o PDF RAG Assistant", divider="gray", anchor=False)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = initialize_baseline_db()
        st.session_state["baseline_docs"] = get_baseline_documents()

    # Display baseline knowledge base info
    with col1.expander("📖 Baseline Knowledge Base"):
        st.write("The following documents are part of the baseline knowledge:")
        for doc in st.session_state.get("baseline_docs", []):
            st.write(f"- {doc}")

    # Regular file upload
    file_upload = col1.file_uploader(
        "Upload a PDF file for analysis ↓", 
        type="pdf", 
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if file_upload:
        if file_upload.name != st.session_state.get("current_file"):
            with st.spinner("Processing uploaded PDF..."):
                # Create vector DB from uploaded file and merge with baseline
                uploaded_db = create_vector_db(file_upload)
                st.session_state["vector_db"] = uploaded_db
                
                # Extract and store PDF pages for display
                pdf_pages = extract_all_pages_as_images(file_upload)
                st.session_state["pdf_pages"] = pdf_pages
                st.session_state["current_file"] = file_upload.name

    # Display PDF if pages are available and it's not the baseline
    if ("pdf_pages" in st.session_state and 
        st.session_state["pdf_pages"] and 
        st.session_state.get("current_file") != BASELINE_NAME):
        
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)
        
        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Modified chat input processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            # Remove analysis_file conditional and just process question
                            response = process_question(
                                prompt,
                                st.session_state["vector_db"]
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a baseline PDF file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")


if __name__ == "__main__":
    main()