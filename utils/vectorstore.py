from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(documents, persist_directory="./chroma_db", collection_name="legal_docs"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    return vectorstore


def load_vectorstore(persist_directory="./chroma_db", collection_name="legal_docs"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore