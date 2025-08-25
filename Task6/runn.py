from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_upd import QDrantUpdater
import sys

sys.stdout.reconfigure(encoding="utf-8")
# Инициализация
qdrant_client = QdrantClient("localhost", port=6333)
embedder = SentenceTransformer('BAAI/bge-m3')

# Создание updater
updater = QDrantUpdater(qdrant_client, "starwars", embedder)

# Запуск обновления
stats = updater.update_collection("./knowledge_base")

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