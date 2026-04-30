from pydantic import BaseModel, Field
from typing import List
from utils import prompts
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import logging

logger = logging.getLogger(__name__)


class ModelResponse(BaseModel):
    sub_questions: List[str] = Field(description="List of sub-questions")


def subquery_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    ).with_structured_output(ModelResponse)

    return prompts.subqueries_prompt | model

def build_rag_chain(vectorstore):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    sub_chain = subquery_chain()

    def get_context_with_sources(subqueries):
        """Retrieve documents, deduplicate, and track sources."""
        if not subqueries or not subqueries.sub_questions:
            logger.warning("No sub-questions generated")
            return {"context": "", "sources": []}
        
        all_docs = []
        for q in subqueries.sub_questions:
            try:
                docs = retriever.invoke(q)
                all_docs.extend(docs)
                logger.info(f"Retrieved {len(docs)} docs for subquery: {q[:50]}")
            except Exception as e:
                logger.warning(f"Retrieval failed for subquery '{q}': {e}")
                continue
        
        if not all_docs:
            logger.warning("No documents retrieved for any subquery")
            return {"context": "", "sources": []}
        
        seen = {}
        unique_docs = []
        for doc in all_docs:
            key = (doc.page_content, doc.metadata.get("source", "unknown"))
            if key not in seen:
                seen[key] = doc
                unique_docs.append(doc)
        
        logger.info(f"Retrieved {len(all_docs)} docs, {len(unique_docs)} unique")
        
        max_tokens = 4000
        token_count = 0
        context_parts = []
        sources = []
        
        for doc in unique_docs:
            doc_tokens = len(doc.page_content.split())
            if token_count + doc_tokens > max_tokens:
                logger.warning(f"Context limit reached. Using {len(context_parts)} docs out of {len(unique_docs)}")
                break
            
            context_parts.append(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)
            token_count += doc_tokens
        
        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources
        }

    main_chain = (
        RunnableMap({
            "subqueries": sub_chain,
            "query": RunnablePassthrough()
        })
        | RunnableLambda(lambda x: {
            **get_context_with_sources(x["subqueries"]),
            "query": x["query"]
        })
        | prompts.main_prompt
        | model
        | StrOutputParser()
    )

    return main_chain