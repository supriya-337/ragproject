import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# 1. Load API keys and setup directories
load_dotenv()

DATA_DIR = "./data"
DB_DIR = "./chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Configure embedding model and LLM
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
llm = ChatHuggingFace(llm=llm_endpoint)

def get_vectorstore():
    """Initializes or connects to the Chroma vector store."""
    return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

def process_and_store_pdf(file_path: str):
    """Loads a PDF, splits it into chunks, and saves to ChromaDB."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)

def delete_file_and_embeddings(file_name: str):
    """Deletes the PDF from disk and removes its chunks from ChromaDB."""
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Remove from disk
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove matching vectors from ChromaDB collection
    vectorstore = get_vectorstore()
    try:
        vectorstore._collection.delete(where={"source": file_path})
    except Exception:
        pass

def get_all_uploaded_files():
    """Returns a list of all PDFs currently saved in the data folder."""
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

def search_and_generate(query: str):
    """Retrieves relevant chunks and generates an answer with citations."""
    vectorstore = get_vectorstore()

    try:
        if vectorstore._collection.count() == 0:
            return "No documents found in the database. Please upload a PDF first."
    except Exception:
        return "No documents found in the database. Please upload a PDF first."

    relevant_docs = vectorstore.similarity_search(query, k=3)
    if not relevant_docs:
        return "I couldn't find anything relevant in your uploaded documents."

    context = ""
    citations = []
    for i, doc in enumerate(relevant_docs):
        context += f"\n\n[Excerpt {i+1}]: {doc.page_content}"
        source_name = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        page_num = doc.metadata.get("page", 0) + 1
        citations.append(f"{source_name} (Page {page_num})")

    prompt = f"""Use the following context excerpts to answer the question. If you cannot find the answer, reply "I cannot answer this based on the provided documents." Answer in the language the question was asked in.

Context: {context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    unique_citations = list(set(citations))
    sources_text = "\n".join([f"- {c}" for c in unique_citations])
    
    return f"{response.content.strip()}\n\n**Sources:**\n{sources_text}"