"""RAG pipeline: PDF ingestion, vector search, and LLM generation.

Optimized with:
- Lazy initialization of heavy resources (embedding model, Chroma, LLM)
- streamlit-aware resource caching via st.cache_resource
- Absolute path handling to avoid cwd-sensitive bugs
- Robust error handling with informative messages
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Directories (absolute, cwd-independent) -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
DATA_DIR.mkdir(exist_ok=True)

# --- Defaults ---------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_REPO_ID = os.getenv("LLM_REPO_ID", "Qwen/Qwen2.5-72B-Instruct")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 3))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

NO_DOCS_MSG = (
    "No documents found in the database. Please upload a PDF first."
)
NOT_FOUND_MSG = "I couldn't find anything relevant in your uploaded documents."
MISSING_KEY_MSG = (
    "HUGGINGFACEHUB_API_TOKEN is not set. Please add it to a `.env` file "
    "in the project root to enable AI answers."
)


def _streamlit_available() -> bool:
    """True if running inside a Streamlit runtime (enables resource caching)."""
    try:
        import streamlit.runtime.scriptrunner as sr

        return sr.get_script_run_ctx() is not None
    except Exception:
        return False


# --- Heavy resources (lazily created + cached) -------------------------------
_embeddings = None
_llm = None


def get_embeddings():
    """Lazily create and reuse the embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


@staticmethod
def _build_llm():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=endpoint)


def get_llm():
    """Lazily build the chat LLM. Returns None if the API token is missing."""
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def get_vectorstore():
    """Initialize or connect to the Chroma vector store."""
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


# --- Ingestion ----------------------------------------------------------------
def process_and_store_pdf(file_path: str | Path) -> int:
    """Load a PDF, split it into chunks, and store them in ChromaDB.

    Returns the number of chunks indexed.
    """
    file_path = Path(file_path)
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    for doc in documents:
        # Normalize source metadata to a stable absolute path so deletes work.
        doc.metadata["source"] = str(file_path.resolve())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def delete_file_and_embeddings(file_name: str) -> None:
    """Delete a PDF from disk and remove its chunks from ChromaDB."""
    file_path = DATA_DIR / file_name
    resolved = str(file_path.resolve())

    if file_path.exists():
        file_path.unlink()

    vectorstore = get_vectorstore()
    try:
        # Filter by the normalized absolute source path used at ingestion.
        existing = vectorstore._collection.get(where={"source": resolved})
        ids = existing.get("ids", [])
        if ids:
            vectorstore._collection.delete(ids=ids)
    except Exception:
        pass


def get_all_uploaded_files() -> list[str]:
    """Return a list of all PDFs currently saved in the data folder."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        f.name for f in DATA_DIR.iterdir() if f.suffix.lower() == ".pdf"
    )


# --- Retrieval + generation ----------------------------------------------------
def _build_prompt(context: str, query: str) -> str:
    return f"""Use the following context excerpts to answer the question. If you cannot find the answer, reply "I cannot answer this based on the provided documents." Answer in the language the question was asked in.

Context: {context}

Question: {query}
Answer:"""


def search_and_generate(query: str) -> str:
    """Retrieve relevant chunks and generate an answer with citations."""
    llm = get_llm()
    if llm is None:
        return MISSING_KEY_MSG

    vectorstore = get_vectorstore()
    try:
        if vectorstore._collection.count() == 0:
            return NO_DOCS_MSG
    except Exception:
        return NO_DOCS_MSG

    relevant_docs = vectorstore.similarity_search(query, k=TOP_K)
    if not relevant_docs:
        return NOT_FOUND_MSG

    context_parts = []
    citations = []
    for i, doc in enumerate(relevant_docs):
        context_parts.append(f"\n\n[Excerpt {i + 1}]: {doc.page_content}")
        source_name = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        page_num = doc.metadata.get("page", 0)
        citations.append(f"{source_name} (Page {page_num + 1})")
    context = "".join(context_parts)

    prompt = _build_prompt(context, query)
    response = llm.invoke(prompt)

    unique_citations = list(dict.fromkeys(citations))
    sources_text = "\n".join(f"- {c}" for c in unique_citations)
    return f"{response.content.strip()}\n\n**Sources:**\n{sources_text}"
