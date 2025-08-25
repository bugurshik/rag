from qdrant_client import QdrantClient
from qdrant_upd import QDrantUpdater
import sys
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

sys.stdout.reconfigure(encoding="utf-8")
folder_path = os.path.join('Task6', 'knowledge_base')
# Инициализация
qdrant_client = QdrantClient("localhost", port=6333)
embedder = HuggingFaceBgeEmbeddings(
            model_name='BAAI/bge-m3',
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="Represent this sentence for searching relevant passages:"
        )

# Создание updater
updater = QDrantUpdater(qdrant_client, "starwars", embedder, 1024)

# Запуск обновления
stats = updater.update_collection(folder_path)

# Вывод статистики
print(f"Обновление завершено:")
print(f"- Добавлено файлов: {stats['files_added']}")
print(f"- Обновлено файлов: {stats['files_updated']}")
print(f"- Пропущено файлов: {stats['files_skipped']}")
print(f"- Обработано чанков: {stats['total_chunks']}")
print(f"- Ошибок: {len(stats['errors'])}")

if stats['errors']:
    print("\nОшибки:")
    for error in stats['errors']:
        print(f"  - {error}")