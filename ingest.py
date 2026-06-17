import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_document(file_path: str, collection_name: str = "documents"):
    print(f"📄 Loading document: {file_path}")
    
    # 1. LOAD
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # 2. CHUNK
    print("✂️  Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")
    
    # 3. EMBED + STORE
    print("🧮 Creating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    print("✅ Document ingested successfully!")
    return vectorstore

if __name__ == "__main__":
    file_path = input("Enter path to your document (.txt file): ")
    ingest_document(file_path)