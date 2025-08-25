import os
from files_loader import loadFiles
from datetime import datetime
from dense_search import BgeDenseSearch

# Конфигурация
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
QDRANT_URL = os.environ.get('QDRANT_URL')
KNOWLEDGE_BASE_DIR = os.environ.get('MD_DIR')
COLLECTION_NAME = f"starwars"

search_engine = BgeDenseSearch(
    model_name= EMBEDDING_MODEL,
    qdrant_url= QDRANT_URL,
    collection_name= COLLECTION_NAME,
    recreate=True  # Удалит старую коллекцию
)

date_start = datetime.now()
print(f"Embedding date start: {date_start}")
documents = loadFiles(KNOWLEDGE_BASE_DIR)
print(documents[0].metadata)
search_engine.add_docs(documents)
date_end = datetime.now()

print(f"Embedding date end: {datetime.now()}")

print(f"{search_engine.count()} чанков сохранено в Qdrant. Операция выполнена за {int((date_end - date_start).total_seconds())} секунд")

testRequest = "События После Большой Уборки"
print(f"Тестовый запрос: '{testRequest}'")
answers = search_engine.search("События После Большой Уборки")
print(answers)