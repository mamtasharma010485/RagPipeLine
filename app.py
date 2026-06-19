# Main RAG application
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from config import *

app = FastAPI(
    title="Production RAG API"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

llm = Ollama(
    model=OLLAMA_MODEL
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {
        "status":"running"
    }

@app.post("/ask")
def ask_question(
    request: QueryRequest
):
    try:
        docs = db.similarity_search(
            request.question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an enterprise assistant.

Answer ONLY using the context.

Context:
{context}

Question:
{request.question}
"""

        answer = llm.invoke(
            prompt
        )

        return {
            "question": request.question,
            "answer": answer,
            "sources": len(docs)
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "error": str(e),
            "traceback": tb
        }