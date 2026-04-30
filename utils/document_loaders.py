from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, WebBaseLoader

def load_pdf_documents(path="./legal-data"):
    loader = DirectoryLoader(
        path=path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    return loader.load()

def load_web_documents(url):
    loader = WebBaseLoader(web_path=url)
    return loader.load()