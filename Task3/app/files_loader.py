from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
import os
import hashlib

def get_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def loadFiles(dir: str):
    print(f"загрузка файлов из {dir}")
    loader = DirectoryLoader(
        dir,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    documents = loader.load()

    # Добавить metadata
    for doc in documents:
        source = doc.metadata["source"]
        doc.metadata["file_hash"] = get_file_hash(source)
        doc.metadata["category"] = os.path.basename(os.path.dirname(source))
        doc.metadata["filename"] = os.path.basename(source)
        doc.metadata["language"] = "ru"
        doc.metadata["format"] = "markdown"

    return documents