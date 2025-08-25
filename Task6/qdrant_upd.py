import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from utils import scan_directory, load_files
import sys
from langchain_core.documents import Document
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client.http.models import Distance, VectorParams

class QDrantUpdater:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embedder:HuggingFaceBgeEmbeddings, vector_size: int = 1024):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.model = embedder
        self.processed_files = set()
        self.splitter =  RecursiveCharacterTextSplitter(
            chunk_size=200,           # длина чанка
            chunk_overlap=20,         # перекрытие между чанками
            length_function=len,      # можно заменить на токенизатор
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        collection_exists = self.client.collection_exists(collection_name)
        if collection_exists == False:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Коллекция '{self.collection_name}' создана с размером вектора {vector_size}.")
            

        sys.stdout.reconfigure(encoding='utf-8')
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format=u'%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('Task6/index_update.log', 'w', 'utf-8'),
                logging.StreamHandler()
            ]
        )

    def get_existing_files_hashes(self) -> Dict[str, str]:
        """Получает информацию о уже обработанных файлах из QDrant"""
        existing_files = {}
        
        try:
            # Получаем все точки с информацией о source
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=10000
            )
            
            for point in scroll_result[0]:
                if 'source' in point.payload and 'file_hash' in point.payload:
                    source = point.payload['source']
                    file_hash = point.payload['file_hash']
                    existing_files[source] = file_hash
                    
        except Exception as e:
            logging.error(f"Ошибка при получении данных из QDrant: {e}")
        
        return existing_files
    
    def delete_file_points(self, file_path: str) -> bool:
        """Удаляет все точки, связанные с файлом"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=file_path)
                        )
                    ]
                )
            )
            logging.info(f"Удалены точки для файла: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Ошибка при удалении точек для {file_path}: {e}")
            return False
    
    def add_doc(self, doc:Document) -> Tuple[int, List[str]]:
                # Генерируем embeddings для документа
                errors = []
                chunks_processed = 0
                try:
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
                    chunks_processed = len(doc_points)
                    logging.info(f"Обработан файл {doc.metadata['source']}: {chunks_processed} чанков")
                except Exception as e:
                    error_msg = f"Ошибка при обработке файлов {doc.metadata['source']}: {e}"
                    errors.append(error_msg)
                    print(f"{e} error")

                return chunks_processed, errors

    def update_collection(self, directory: str) -> Dict[str, int]:
        """Основной метод для обновления коллекции"""
        start_time = datetime.now()
        stats = {
            'files_added': 0,
            'files_updated': 0,
            'files_skipped': 0,
            'errors': [],
            'total_chunks': 0
        }
        
        try:
            # Сканируем директорию
            current_files = load_files(directory)
            logging.info(f"Found {len(current_files)} files in directory: {directory}")
            
            # Получаем информацию о уже обработанных файлах
            existing_files = self.get_existing_files_hashes()
            logging.info(f"Found {len(existing_files)} exist files in QDrant")
            
            # Обрабатываем каждый файл
            for i,doc in enumerate(current_files):
                file_path = doc.metadata['source']
                current_hash = doc.metadata['file_hash']

                if file_path in existing_files:
                    # Файл уже существует, проверяем хеш
                    
                    if existing_files[file_path] == current_hash:
                        # Файл не изменился
                        stats['files_skipped'] += 1
                        logging.info(f"No changes - Skip: {file_path}")
                        continue
                    else:
                        # Файл изменился - удаляем старые точки и обрабатываем заново
                        if self.delete_file_points(file_path):
                            chunks_processed, errors = self.add_doc(doc)
                            stats['files_updated'] += 1
                            stats['total_chunks'] += chunks_processed
                            stats['errors'].extend(errors)
                else:
                    # Новый файл
                    chunks_processed, errors = self.add_doc(doc)
                    stats['files_added'] += 1
                    stats['total_chunks'] += chunks_processed
                    stats['errors'].extend(errors)

            # Записываем итоговый лог
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            log_message = (
                f"index updated at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"{stats['files_added']} files added, "
                f"{stats['files_updated']} files updated, "
                f"{stats['files_skipped']} files skipped, "
                f"{stats['total_chunks']} chunks processed, "
                f"{len(stats['errors'])} errors, "
                f"duration: {duration:.2f}s"
            )
            
            logging.info(log_message)
            print(log_message)
            
        except Exception as e:
            error_msg = f"Error: {e}"
            logging.error(error_msg)
            stats['errors'].append(error_msg)
        
        return stats