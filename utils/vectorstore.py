import logging
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)


def create_vectorstore(documents, persist_directory="./chroma_db", collection_name="legal_docs"):
    """Create and persist vectorstore from documents."""
    try:
        if not documents:
            logger.error("No documents provided to create vectorstore")
            raise ValueError("Empty document list")
        
        logger.info(f"Creating vectorstore with {len(documents)} documents")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        logger.info(f"Vectorstore created successfully with {len(documents)} documents")
        return vectorstore
    
    except Exception as e:
        logger.error(f"Failed to create vectorstore: {e}", exc_info=True)
        raise


def load_vectorstore(persist_directory="./chroma_db", collection_name="legal_docs"):
    """Load existing vectorstore from disk."""
    try:
        logger.info(f"Loading vectorstore from {persist_directory}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        doc_count = vectorstore._collection.count()
        if doc_count == 0:
            logger.warning("Vectorstore is empty")
            return None
        
        logger.info(f"Vectorstore loaded successfully with {doc_count} documents")
        return vectorstore
    
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {e}", exc_info=True)
        return None