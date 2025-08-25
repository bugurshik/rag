from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Any
import uuid
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointsSelector

class BgeDenseSearch:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "dense_collection",
        vector_size: int = 1024,
        recreate: bool = False
    ):
        """
        Инициализация системы семантического поиска.

        :param model_name: Название модели для эмбеддингов (например, BAAI/bge-m3)
        :param qdrant_url: URL Qdrant сервера
        :param collection_name: Имя коллекции в Qdrant
        :param vector_size: Размер вектора
        :param recreate: Пересоздать коллекцию при запуске
        """
        self.model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="Represent this sentence for searching relevant passages:"
        )
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.splitter =  RecursiveCharacterTextSplitter(
            chunk_size=200,           # длина чанка
            chunk_overlap=20,         # перекрытие между чанками
            length_function=len,      # можно заменить на токенизатор
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        if recreate:
            self._create_collection()

    def _create_collection(self):
        """Создаёт коллекцию в Qdrant."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print(f"Коллекция '{self.collection_name}' создана с размером вектора {self.vector_size}.")

    def add_docs(self, docs: list[Document]):
        try:
            for i,doc in enumerate(docs):
                # Генерируем embeddings для документа
                source = doc.metadata["source"]
                print(f"start embedding document {source}")

                doc_points = []
                chunks = self.splitter.split_text(doc.page_content)
                for y, chunk in enumerate(chunks):
                    doc_points.append(models.PointStruct(
                        id= str(uuid.uuid4()),
                        vector=self.model.embed_query(chunk),
                        payload={
                            "page_content": chunk,
                            "chunk_id": y,
                            **doc.metadata  # все метаданные добавляются сюда
                        }
                    ))

                # Вставляем точки
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=doc_points
                )
                print(f"upsert document chunks {source} successfully")

        except Exception as e:
            print(f"{e} error")

    def get_all_hashes_in_qdrant(self) -> set[str]:
        """
        Получает ВСЕ file_hash из коллекции Qdrant с использованием пагинации.
        """
        hashes: set[str] = set()
        offset = None
        seen_points = 0
        limit = 1000  # Размер страницы (разумный баланс между кол-вом запросов и памятью)

        print(" Начинаем получение всех хешей из Qdrant...")

        while True:
            # Получаем порцию точек
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,  # Все точки
                with_payload=True,
                with_vectors=False,
                limit=limit,
                offset=offset,
                order_by=None 
            )

            points, next_offset = response

            # Извлекаем хеши
            batch_hashes = {
                point.payload.get("file_hash")
                for point in points
                if point.payload and point.payload.get("file_hash")
            }
            hashes.update({h for h in batch_hashes if h})  # Исключаем None

            seen_points += len(points)
            print(f"  → Загружено {seen_points} точек... (уникальных хешей: {len(hashes)})")

            # Условие выхода
            if next_offset is None:
                break

            offset = next_offset

        print(f"Завершено. Всего найдено {len(hashes)} уникальных file_hash")
        return hashes

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Выполняет поиск по запросу.

        :param query: Поисковый запрос
        :param limit: Количество результатов
        :return: Список результатов с текстом и оценкой
        """
        query_vector = self.model.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text"),
                "payload": hit.payload
            }
            for hit in results
        ]

    def count(self) -> int:
        """Возвращает количество документов в коллекции."""
        count_response = self.client.count(collection_name=self.collection_name)
        return count_response.count