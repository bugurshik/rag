from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

class FilteringRetriever(BaseRetriever):
    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Получаем документы из оригинального retriever'а
        docs = self.retriever.get_relevant_documents(query)
        
        # Фильтруем нежелательные паттерны
        filtered_docs = []
        for doc in docs:
            content = doc.page_content
            
            # Проверяем на наличие опасных паттернов
            if self.contains_malicious_pattern(content):
                continue  # пропускаем
            filtered_docs.append(doc)
        
        return filtered_docs

    def contains_malicious_pattern(self, text: str) -> bool:
        import re
        patterns = [
            r"(?i)ignore all (instructions|rules)",
            r"(?i)disregard the (above|previous)",
            r"(?i)you are now (GPT|AI) free",
            r"(?i)system prompt",
            r"(?i)forget all previous instructions",
        ]
        return any(re.search(pattern, text) for pattern in patterns)