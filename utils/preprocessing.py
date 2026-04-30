import re
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def preprocess_pdfs(pdf_docs):
    """Preprocess PDF documents for RAG pipeline."""
    if not pdf_docs:
        logger.warning("No PDF documents to preprocess")
        return []
    
    processed_docs = []
    
    for doc in pdf_docs:
        try:
            # Preserve metadata from PDF
            metadata = doc.metadata or {}
            metadata["type"] = "pdf"
            metadata["source"] = metadata.get("source", "unknown_pdf")
            
            text = doc.page_content.strip()
            
            # Generic cleaning for legal PDFs
            # Remove excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Remove page breaks and markers
            text = re.sub(r'(?:---+|===+|\*\*\*+)', '', text)
            
            # Remove common footer/header patterns
            text = re.sub(r'(Page\s+\d+|©.*?(?:\n|$)|CONFIDENTIAL|DRAFT)', '', text, flags=re.IGNORECASE | re.MULTILINE)
            
            # Remove multiple spaces but preserve line breaks
            text = re.sub(r' {2,}', ' ', text)
            
            # Remove isolated special characters
            text = re.sub(r'[\^\*\~]{2,}', '', text)
            
            # Clean up
            text = text.strip()
            
            # Skip documents that are too short
            if len(text.split()) < 50:
                logger.warning(f"PDF document too short after preprocessing: {metadata.get('source')}")
                continue
            
            processed_docs.append(Document(page_content=text, metadata=metadata))
            logger.info(f"Processed PDF: {metadata.get('source')} ({len(text.split())} words)")
        
        except Exception as e:
            logger.error(f"Error preprocessing PDF document: {e}")
            continue
    
    return processed_docs


def preprocess_webdocs(webdocs):
    """Preprocess web documents for RAG pipeline."""
    if not webdocs:
        logger.warning("No web documents to preprocess")
        return []
    
    processed_docs = []
    
    for doc in webdocs:
        try:
            metadata = doc.metadata or {}
            metadata["type"] = "web"
            metadata["source"] = metadata.get("source", "unknown_web")
            
            text = doc.page_content.strip()
            
            # Generic cleaning for web content
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'(TAGS|Tags|tags|footnotes|Footnotes|©.*?$)', '', text, flags=re.MULTILINE)
            text = re.sub(r'[-—]\s*\d+\s*[-—]', '', text)
            text = re.sub(r' {2,}', ' ', text)
            
            text = text.strip()
            
            if len(text.split()) < 100:
                logger.warning(f"Web document too short after preprocessing: {metadata.get('source')}")
                continue
            
            processed_docs.append(Document(page_content=text, metadata=metadata))
            logger.info(f"Processed web doc: {metadata.get('source')} ({len(text.split())} words)")
        
        except Exception as e:
            logger.error(f"Error preprocessing web document: {e}")
            continue
    
    return processed_docs


def preprocess_documents(documents):
    """Unified preprocessing for mixed document types."""
    if not documents:
        return []
    
    pdf_docs = [doc for doc in documents if doc.metadata.get("source", "").endswith(".pdf")]
    web_docs = [doc for doc in documents if not doc.metadata.get("source", "").endswith(".pdf")]
    
    processed = []
    processed.extend(preprocess_pdfs(pdf_docs))
    processed.extend(preprocess_webdocs(web_docs))
    
    logger.info(f"Total preprocessed documents: {len(processed)}")
    return processed