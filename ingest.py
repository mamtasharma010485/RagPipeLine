# Document ingestion script
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import *

def load_documents_from_folder(folder_path="knowledge"):
    """Load all PDF and DOCX files from the knowledge folder"""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"Error: {folder_path} folder not found")
        return documents
    
    # Get all files in the knowledge folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        try:
            if file.lower().endswith('.pdf'):
                print(f"Loading PDF: {file}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                
            elif file.lower().endswith('.docx'):
                print(f"Loading DOCX: {file}")
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                
            else:
                print(f"Skipping unsupported file: {file}")
                
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

# Load all documents from knowledge folder
documents = load_documents_from_folder()

if not documents:
    print("No documents found. Please add PDF or DOCX files to the knowledge folder.")
else:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    print("Knowledge Base Created Successfully!")