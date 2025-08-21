from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from qdrant_client import QdrantClient
from tqdm import tqdm
import os

# Конфигурация
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
QDRANT_URL = os.environ.get('QDRANT_URL')
KNOWLEDGE_BASE_DIR = os.environ.get('MD_DIR')


COLLECTION_NAME = f"{KNOWLEDGE_BASE_DIR}:{EMBEDDING_MODEL}:embeddings"

print(f"{COLLECTION_NAME} загрузка модели")
# 1. Загрузка модели
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={'normalize_embeddings': True},
    query_instruction="Represent this sentence for searching relevant passages:"
)

print(f"{COLLECTION_NAME} загрузка файлов")
# 2. Загрузка .md файлов
loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()

print(f"{COLLECTION_NAME} Создание")
# 3. Создание и сохранение эмбедингов
Qdrant.from_documents(
    documents,
    embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True,
    vector_name="dense"
)

print(f"Эмбединги для {len(documents)} документов сохранены в Qdrant")
