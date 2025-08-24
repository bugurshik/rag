import os
import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition

# Configuration
CONFIG = {
    "source_dir": "./knowledge_base",           # Папка с документами
    "processed_files": "./processed_files.json",  # Файл с историей обработки
    "vector_db_path": "./chroma_db",  # Путь к векторной БД
    "log_file": "./update_index.log", # Файл логов
    "embedding_model": "BAAI/bge-m3",
    "chunk_size": 200,
    "chunk_overlap": 20,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeBaseUpdater:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = SentenceTransformer(config['embedding_model'])
        self.chroma_client = chromadb.PersistentClient(path=config['vector_db_path'])
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self) -> dict:
        """Загружает историю обработанных файлов"""
        if os.path.exists(self.config['processed_files']):
            with open(self.config['processed_files'], 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        """Сохраняет историю обработанных файлов"""
        with open(self.config['processed_files'], 'w') as f:
            json.dump(self.processed_files, f, indent=2)

    def _get_file_hash(self, file_path: str) -> str:
        """Вычисляет хеш файла для отслеживания изменений"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _find_new_files(self) -> list:
        """Находит новые или измененные файлы"""
        new_files = []
        source_path = Path(self.config['source_dir'])
        
        if not source_path.exists():
            logger.error(f"Source directory {self.config['source_dir']} does not exist!")
            return new_files

        for file_path in source_path.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.html']:
                file_hash = self._get_file_hash(str(file_path))
                relative_path = str(file_path.relative_to(source_path))
                
                # Проверяем, новый файл или измененный
                if (relative_path not in self.processed_files or 
                    self.processed_files[relative_path]['hash'] != file_hash):
                    
                    new_files.append({
                        'path': str(file_path),
                        'relative_path': relative_path,
                        'hash': file_hash,
                        'size': file_path.stat().st_size
                    })
                    logger.info(f"Found new/modified file: {relative_path}")
        
        return new_files

    def _process_file(self, file_info: dict) -> list:
        """Обрабатывает файл: извлекает текст и разбивает на чанки"""
        try:
            # Извлекаем текст из файла
            elements = partition(filename=file_info['path'])
            text_content = "\n\n".join([str(el) for el in elements])
            
            # Разбиваем на чанки
            chunks = []
            for i in range(0, len(text_content), self.config['chunk_size'] - self.config['chunk_overlap']):
                chunk = text_content[i:i + self.config['chunk_size']]
                if chunk.strip():
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source_file': file_info['relative_path'],
                            'chunk_index': len(chunks),
                            'file_size': file_info['size'],
                            'processed_at': datetime.now().isoformat()
                        }
                    })
            
            logger.info(f"Processed {file_info['relative_path']} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_info['path']}: {e}")
            return []

    def _update_vector_db(self, chunks: list):
        """Обновляет векторную базу данных"""
        if not chunks:
            return

        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"{metadata['source_file']}_{metadata['chunk_index']}" for metadata in metadatas]
        
        # Генерируем эмбеддинги
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Добавляем в коллекцию
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector database")

    def run_update(self):
        """Основной метод обновления базы знаний"""
        start_time = datetime.now()
        logger.info("=" * 50)
        logger.info(f"Starting knowledge base update at {start_time}")
        
        try:
            # Ищем новые файлы
            new_files = self._find_new_files()
            logger.info(f"Found {len(new_files)} new/modified files")
            
            total_chunks = 0
            processed_count = 0
            
            # Обрабатываем каждый файл
            for file_info in new_files:
                chunks = self._process_file(file_info)
                if chunks:
                    self._update_vector_db(chunks)
                    total_chunks += len(chunks)
                    processed_count += 1
                    
                    # Обновляем историю обработки
                    self.processed_files[file_info['relative_path']] = {
                        'hash': file_info['hash'],
                        'processed_at': datetime.now().isoformat(),
                        'chunks_count': len(chunks)
                    }
            
            # Сохраняем историю обработки
            self._save_processed_files()
            
            # Получаем статистику
            total_documents = self.collection.count()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Логируем результаты
            logger.info(f"Update completed at {end_time}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Files processed: {processed_count}/{len(new_files)}")
            logger.info(f"Total chunks added: {total_chunks}")
            logger.info(f"Total documents in index: {total_documents}")
            logger.info("=" * 50)
            
            return {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'files_processed': processed_count,
                'chunks_added': total_chunks,
                'total_documents': total_documents,
                'errors': 0
            }
            
        except Exception as e:
            logger.error(f"Update failed with error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'start_time': start_time.isoformat()
            }

def main():
    """Точка входа"""
    updater = KnowledgeBaseUpdater(CONFIG)
    result = updater.run_update()
    
    # Выводим краткий отчет в stdout
    if result['status'] == 'success':
        print(f"Index updated at {result['end_time']}, "
              f"{result['files_processed']} files processed, "
              f"{result['chunks_added']} chunks added, "
              f"total documents: {result['total_documents']}, "
              f"0 errors")
    else:
        print(f"Update failed: {result['error']}")

if __name__ == "__main__":
    main()