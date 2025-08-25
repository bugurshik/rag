import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from utils import scan_directory, load_files
import sys

class QDrantUpdater:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embedder):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder
        self.processed_files = set()

        sys.stdout.reconfigure(encoding='utf-8')
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format=u'%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./index_update.log', 'w', 'utf-8'),
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
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Разбивает текст на чанки с перекрытием"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def process_file(self, file_path: str, file_hash: str) -> Tuple[int, List[str]]:
        """Обрабатывает файл: читает, разбивает на чанки, создает эмбеддинги"""
        errors = []
        chunks_processed = 0
        
        try:
            # Чтение файла
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разбивка на чанки
            chunks = self.chunk_text(content)
            
            points = []
            for i, chunk in enumerate(chunks):
                try:
                    # Генерация эмбеддинга
                    embedding = self.embedder.embed_query(chunk)  # Предполагаем, что embedder имеет метод embed_query
                    
                    # Создание точки для QDrant
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": file_path,
                            "file_hash": file_hash,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    error_msg = f"Ошибка при обработке чанка {i} файла {file_path}: {e}"
                    errors.append(error_msg)
                    logging.error(error_msg)
            
            # Добавление точек в QDrant
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                chunks_processed = len(points)
                logging.info(f"Обработан файл {file_path}: {chunks_processed} чанков")
            
        except Exception as e:
            error_msg = f"Ошибка при обработке файла {file_path}: {e}"
            errors.append(error_msg)
            logging.error(error_msg)
        
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
            for file_path, current_hash in current_files.items():
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
                            chunks_processed, errors = self.process_file(file_path, current_hash)
                            stats['files_updated'] += 1
                            stats['total_chunks'] += chunks_processed
                            stats['errors'].extend(errors)
                else:
                    # Новый файл
                    chunks_processed, errors = self.process_file(file_path, current_hash)
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