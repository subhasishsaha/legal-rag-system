import os
import logging
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, WebBaseLoader

logger = logging.getLogger(__name__)

def load_pdf_documents(path="legal-data"):
    """Load PDF documents from directory."""
    try:
        if not os.path.exists(path):
            logger.warning(f"PDF directory not found: {path}")
            return []
        
        pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"No PDF files found in {path}")
            return []
        
        loader = DirectoryLoader(
            path=path,
            glob="*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} PDF documents from {path}")
        return docs
    
    except Exception as e:
        logger.error(f"Error loading PDF documents: {e}")
        return []


def load_web_documents(url):
    """Load documents from web URL."""
    try:
        if not url:
            logger.warning("No URL provided for web document loading")
            return []
        
        loader = WebBaseLoader(web_path=url)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from {url}")
        return docs
    
    except Exception as e:
        logger.error(f"Error loading web documents from {url}: {e}")
        return []