from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from tqdm import tqdm
import os
import uuid

# Конфигурация
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
QDRANT_URL = os.environ.get('QDRANT_URL')
KNOWLEDGE_BASE_DIR = os.environ.get('MD_DIR')

COLLECTION_NAME = f"starwars"

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

# Добавить metadata
for doc in documents:
    source = doc.metadata["source"]
    filename = os.path.basename(source)
    folder = os.path.basename(os.path.dirname(source))

    doc.metadata["category"] = folder
    doc.metadata["filename"] = filename
    doc.metadata["doc_type"] = "markdown_kb"
    doc.metadata["language"] = "ru"
    doc.metadata["format"] = "markdown"

print(documents[0].metadata)

# 3. Настройка разделителя
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,           # длина чанка
    chunk_overlap=20,         # перекрытие между чанками
    length_function=len,      # можно заменить на токенизатор
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 4. Подготовка qdrant
print(f"{COLLECTION_NAME} подключение к Qdrant")
client = QdrantClient(url=QDRANT_URL)

try:
    print(f"start recreate_collection...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

except Exception as e:
     print(f"не удалось пересоздать коллекцию: {e}")


# 5. Загрузка embeddings
try:
    for i,doc in enumerate(documents):
        # Генерируем embeddings для документа
        source = doc.metadata["source"]
        print(f"start embedding document {source}")

        text = doc.page_content
        doc_points = []
        chunks = text_splitter.split_text(text)
        for y, chunk in enumerate(chunks):
            doc_points.append(models.PointStruct(
                id= str(uuid.uuid4()),
                vector=embeddings.embed_query(chunk),
                payload={
                    "page_content": chunk,
                    "chunk_id": y,
                    **doc.metadata  # все метаданные добавляются сюда
                }
            ))

        # Вставляем точки
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=doc_points
        )
        print(f"upsert document chunks {source} successfully")

except Exception as e:
    print(f"{e} error")

print(f"Эмбединги для {len(documents)} документов сохранены в Qdrant")
