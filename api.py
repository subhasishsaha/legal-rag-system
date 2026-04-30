from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

from utils.preprocessing import preprocess_documents
from utils import document_loaders, text_splitter, vectorstore as vs
from chains.chains import build_rag_chain

app = FastAPI(title="Legal RAG API", version="1.0.0")

CHROMA_DIR = "./chroma_db"
URL = "https://www.scobserver.in/journal/supreme-court-review-top-10-judgements-of-2025/"


# ---- Load everything once at startup ----
def initialize_pipeline():
    try:
        logger.info("Initializing RAG pipeline...")
        
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            logger.info("Loading cached vectorstore")
            vectorstore = vs.load_vectorstore()
        else:
            logger.info("Building vectorstore from documents")
            
            pdf_docs = document_loaders.load_pdf_documents()
            web_docs = document_loaders.load_web_documents(URL)
            
            all_docs = pdf_docs + web_docs
            logger.info(f"Loaded {len(pdf_docs)} PDFs, {len(web_docs)} web docs")
            
            processed_docs = preprocess_documents(all_docs)
            
            if not processed_docs:
                raise ValueError("No documents loaded after preprocessing. Check PDF directory and URL.")
            
            logger.info(f"Preprocessed {len(processed_docs)} documents")
            
            split_docs = text_splitter.split_documents(processed_docs)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            vectorstore = vs.create_vectorstore(split_docs)
        
        qa_chain = build_rag_chain(vectorstore)
        logger.info("RAG pipeline initialized successfully")
        return qa_chain
    
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
        raise

qa_chain = initialize_pipeline()


# ---- Request/Response schema ----
class QueryRequest(BaseModel):
    query: str = Field(..., description="The legal question to be answered")


class QueryResponse(BaseModel):
    answer: str


# ---- API endpoint ----
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Answer legal question using RAG pipeline."""
    if qa_chain is None:
        logger.error("Pipeline not initialized")
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        result = qa_chain.invoke(request.query)
        
        answer = result if isinstance(result, str) else str(result)
        
        if not answer or answer.strip() == "":
            logger.warning("Empty answer returned from chain")
            raise HTTPException(status_code=400, detail="Could not generate answer")
        
        logger.info("Query processed successfully")
        return QueryResponse(answer=answer)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query")
    

@app.get("/health")
def health_check():
    """Health check endpoint - returns pipeline status."""
    if qa_chain is None:
        logger.warning("Health check: Pipeline not initialized")
        return {
            "status": "unhealthy",
            "message": "Pipeline not initialized",
            "code": 503
        }
    
    try:
        return {
            "status": "healthy",
            "message": "RAG pipeline is ready",
            "code": 200
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "message": f"Pipeline error: {str(e)}",
            "code": 500
        }
    
@app.get("/")
def root():
    return {"message": "Welcome to the Legal RAG API. Use /ask to get answers to legal questions.", "version": "1.0.0"}