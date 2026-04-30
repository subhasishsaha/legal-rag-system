from fastapi import FastAPI
from pydantic import BaseModel
import os

from dotenv import load_dotenv

from utils.preprocessing import preprocess_webdocs
load_dotenv()

from utils import document_loaders, text_splitter, vectorstore as vs
from chains.chains import build_rag_chain

app = FastAPI(title="Legal RAG API")

CHROMA_DIR = "./chroma_db"
URL = "https://www.scobserver.in/journal/supreme-court-review-top-10-judgements-of-2025/"


# ---- Load everything once at startup ----
def initialize_pipeline():
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = vs.load_vectorstore()
    else:
        pdf_docs = document_loaders.load_pdf_documents()
        web_docs = document_loaders.load_web_documents(URL)

        docs = pdf_docs + web_docs
        web_docs = preprocess_webdocs(web_docs)

        split_docs = text_splitter.split_documents(web_docs)
        vectorstore = vs.create_vectorstore(split_docs)

    chain = build_rag_chain(vectorstore)
    return chain


qa_chain = initialize_pipeline()


# ---- Request/Response schema ----
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


# ---- API endpoint ----
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    result = qa_chain.invoke(request.query)
    return QueryResponse(answer=result)


# ---- Health check ----
@app.get("/")
def root():
    return {"status": "API is running"}