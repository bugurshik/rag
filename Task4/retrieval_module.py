from qdrant_client import QdrantClient
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class DataRetrieval:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "starwars",
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

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Поиск в Qdrant по эмбеддингам"""
        # Генерируем эмбеддинг запроса
        query_vec = self.model.embed_query(query, normalize_embeddings=True)
        query_vec = query_vec.tolist()

        # Поиск
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=k,
            with_payload=True
        )
        # Извлекаем текст
        documents = []
        for hit in results:
            payload = hit.payload
            # Предполагаем, что в payload есть ключ "text" или "content"
            text = payload.get("text") or payload.get("content") or payload.get("document")
            if text:
                documents.append(text)

        return documents
    
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