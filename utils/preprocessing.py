import re
from langchain_core.documents import Document

def preprocess_webdocs(webdocs):
    doc = webdocs[0]

    metadata = doc.metadata
    metadata["type"] = "landmark judgements"

    text = doc.page_content

    whitespace = r'\n{2,}'
    footers = r'TAGS'
    start_pattern = r'1\. Court has limited power'

    text = re.sub(whitespace, " ", text).strip()

    match = re.search(footers, text)
    if match:
        text = text[:match.start()]
        
    match = re.search(start_pattern, text)
    if match:
        text = text[match.start():]

    return [Document(page_content=text, metadata=metadata)]