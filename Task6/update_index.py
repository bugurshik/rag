import os
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Set
from pathlib import Path
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointsSelector
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
import uuid
from qdrant_client.http import models

class BgeDenseSearchUpdater():
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "starwars",
        vector_size: int = 1024,
        recreate: bool = False
    ):
        """
        Инициализация системы обновления индекса.
        
        :param model_name: Название модели для эмбеддингов
        :param qdrant_url: URL Qdrant сервера
        :param collection_name: Имя коллекции в Qdrant
        :param vector_size: Размер вектора
        :param recreate: Пересоздать коллекцию при запуске
        """
        super().__init__(model_name, qdrant_url, collection_name, vector_size, recreate)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./index_update.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Вычисляет хеш файла для определения изменений.
        
        :param file_path: Путь к файлу
        :return: MD5 хеш файла
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении хеша файла {file_path}: {e}")
            return ""

    def scan_directory(self, directory_path: str, extensions: List[str] = None) -> List[Dict]:
        """
        Сканирует директорию и возвращает информацию о файлах.
        
        :param directory_path: Путь к директории для сканирования
        :param extensions: Список разрешений файлов (например, ['.txt', '.pdf'])
        :return: Список словарей с информацией о файлах
        """
        if extensions is None:
            extensions = ['.md']
        
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Директория {directory_path} не существует")
            return []
        
        files_info = []
        self.logger.info(f"Сканирование директории: {directory_path}")
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    file_hash = self._calculate_file_hash(str(file_path))
                    if file_hash:  # Если хеш успешно вычислен
                        files_info.append({
                            'path': str(file_path),
                            'hash': file_hash,
                            'size': file_path.stat().st_size,
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                        })
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        
        self.logger.info(f"Найдено {len(files_info)} файлов")
        return files_info

    def identify_new_and_modified_files(self, source_directory: str, 
                                      extensions: List[str] = None) -> List[Dict]:
        """
        Идентифицирует новые и изменённые файлы.
        
        :param source_directory: Директория для сканирования
        :param extensions: Список разрешений файлов
        :return: Список новых и изменённых файлов
        """
        # Получаем текущие файлы
        current_files = self.scan_directory(source_directory, extensions)
        
        # Получаем существующие хеши из Qdrant
        existing_hashes = self.get_all_hashes_in_qdrant()
        
        # Идентифицируем новые и изменённые файлы
        new_and_modified = []
        for file_info in current_files:
            if file_info['hash'] not in existing_hashes:
                new_and_modified.append(file_info)
        
        self.logger.info(f"Найдено {len(new_and_modified)} новых/изменённых файлов")
        return new_and_modified

    def load_document(self, file_path: str) -> Document:
        """
        Загружает документ из файла.
        
        :param file_path: Путь к файлу
        :return: Объект Document
        """
        try:
            docs = self.loadFiles(file_path)
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
            self.logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
            return None

    def loadFiles(self, dir: str):
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
            doc.metadata["file_hash"] = self.get_file_hash(source)
            doc.metadata["category"] = os.path.basename(os.path.dirname(source))
            doc.metadata["filename"] = os.path.basename(source)
            doc.metadata["language"] = "ru"
            doc.metadata["format"] = "markdown"

        return documents
    
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

    def remove_document_chunks(self, file_hash: str):
        """
        Удаляет все чанки документа по его хешу.
        
        :param file_hash: Хеш файла для удаления
        """
        try:
            self.logger.info(f"Удаление чанков для файла с хешем: {file_hash}")
            
            # Создаём фильтр для поиска точек по file_hash
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            )
            
            # Находим все точки для удаления
            points_to_delete = []
            offset = None
            limit = 1000
            
            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    with_payload=False,
                    with_vectors=False,
                    limit=limit,
                    offset=offset
                )
                
                points, next_offset = response
                points_to_delete.extend([point.id for point in points])
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # Удаляем точки
            if points_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointsSelector(
                        points=points_to_delete
                    )
                )
                self.logger.info(f"Удалено {len(points_to_delete)} чанков для файла {file_hash}")
            else:
                self.logger.info(f"Не найдено чанков для удаления файла {file_hash}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при удалении чанков файла {file_hash}: {e}")

    def update_index(self, source_directory: str, extensions: List[str] = None, 
                   remove_obsolete: bool = True):
        """
        Основной метод для обновления индекса.
        
        :param source_directory: Директория для сканирования
        :param extensions: Список разрешений файлов
        :param remove_obsolete: Удалять ли устаревшие документы
        """
        self.logger.info("=" * 50)
        self.logger.info("НАЧАЛО ОБНОВЛЕНИЯ ИНДЕКСА")
        self.logger.info("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Шаг 1: Идентификация новых и изменённых файлов
            files_to_process = self.identify_new_and_modified_files(source_directory, extensions)
            
            if not files_to_process:
                self.logger.info("Нет новых или изменённых файлов для обработки")
                return
            
            # Шаг 2: Обработка каждого файла
            processed_count = 0
            for file_info in files_to_process:
                try:
                    self.logger.info(f"Обработка файла: {file_info['path']}")
                    
                    # Удаляем старые чанки (если файл изменён)
                    self.remove_document_chunks(file_info['file_hash'])
                    
                    # Загружаем и обрабатываем документ
                    doc = self.load_document(file_info['path'])
                    if doc:
                        self.add_docs([doc])
                        processed_count += 1
                        self.logger.info(f"Файл успешно обработан: {file_info['path']}")
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке файла {file_info['path']}: {e}")
                    continue
            
            # Шаг 3: Удаление устаревших документов (опционально)
            if remove_obsolete:
                self._remove_obsolete_documents(source_directory, extensions)
            
            # Логирование результатов
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 50)
            self.logger.info("ЗАВЕРШЕНИЕ ОБНОВЛЕНИЯ ИНДЕКСА")
            self.logger.info(f"Обработано файлов: {processed_count}/{len(files_to_process)}")
            self.logger.info(f"Общее время: {duration}")
            self.logger.info(f"Всего документов в индексе: {self.count()}")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка при обновлении индекса: {e}")

    def _remove_obsolete_documents(self, source_directory: str, extensions: List[str] = None):
        """
        Удаляет документы, которые больше не существуют в источнике.
        
        :param source_directory: Директория источника
        :param extensions: Список разрешений файлов
        """
        try:
            self.logger.info("Поиск устаревших документов для удаления...")
            
            # Получаем текущие файлы в источнике
            current_files = self.scan_directory(source_directory, extensions)
            current_hashes = {file_info['file_hash'] for file_info in current_files}
            
            # Получаем все хеши из Qdrant
            qdrant_hashes = self.get_all_hashes_in_qdrant()
            
            # Находим хеши, которые есть в Qdrant, но нет в источнике
            obsolete_hashes = qdrant_hashes - current_hashes
            
            if obsolete_hashes:
                self.logger.info(f"Найдено {len(obsolete_hashes)} устаревших документов")
                
                # Удаляем устаревшие документы
                for file_hash in obsolete_hashes:
                    self.remove_document_chunks(file_hash)
                
                self.logger.info(f"Удалено {len(obsolete_hashes)} устаревших документов")
            else:
                self.logger.info("Устаревшие документы не найдены")
                
        except Exception as e:
            self.logger.error(f"Ошибка при удалении устаревших документов: {e}")

if __name__ == "__main__":
    updater = BgeDenseSearchUpdater(
        model_name="BAAI/bge-m3",
        qdrant_url="http://localhost:6333",
        collection_name="starwars",
        recreate=False
    )
    
    # Обновление индекса
    updater.update_index(
        source_directory="./knowledge_base",
        extensions=['.md'],
        remove_obsolete=True
    )