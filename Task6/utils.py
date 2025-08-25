import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

def compute_file_hash(file_path: str) -> str:
    """Вычисляет хеш файла для отслеживания изменений"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def scan_directory(directory: str, extensions: List[str] = None) -> Dict[str, str]:
    """
    Сканирует директорию и возвращает словарь файлов с их хешами
    """
    if extensions is None:
        extensions = ['.md', '.txt']
    
    file_hashes = {}
    directory_path = Path(directory)
    
    for ext in extensions:
        for file_path in directory_path.rglob(f"*{ext}"):
            if file_path.is_file():
                try:
                    file_hash = compute_file_hash(str(file_path))
                    file_hashes[str(file_path)] = file_hash
                except Exception as e:
                    logging.error(f"Ошибка при чтении файла {file_path}: {e}")
    
    return file_hashes

def cleanup_old_files(self, directory: str):
    """Удаляет точки для файлов, которых больше нет в директории"""
    current_files = set(scan_directory(directory).keys())
    existing_files = set(self.get_existing_files_hashes().keys())
    
    files_to_remove = existing_files - current_files
    
    for file_path in files_to_remove:
        self.delete_file_points(file_path)
        logging.info(f"Удален отсутствующий файл: {file_path}")
    
    return len(files_to_remove)

def get_collection_stats(self):
    """Получает статистику коллекции"""
    try:
        info = self.client.get_collection(self.collection_name)
        count = self.client.count(self.collection_name)
        return {
            'vectors_count': info.vectors_count,
            'points_count': count.count,
            'status': info.status
        }
    except Exception as e:
        logging.error(f"Ошибка при получении статистики: {e}")
        return None

def load_files(dir: str) -> List[Document]:
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
        doc.metadata["file_hash"] = compute_file_hash(source)
        doc.metadata["category"] = os.path.basename(os.path.dirname(source))
        doc.metadata["filename"] = os.path.basename(source)
        doc.metadata["language"] = "ru"
        doc.metadata["format"] = "markdown"

    return documents