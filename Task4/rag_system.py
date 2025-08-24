from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from qdrant_client import QdrantClient

class RAGSystem:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model_name: str = "BAAI/bge-m3",
        llm_model_name: str = "Qwen/Qwen1.5-1.8B",
        prompt_template: str = None,
        device: str = "cuda"
    ):
        """
        Инициализация RAG системы.
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.device = device

        # --- 1. Эмбеддинги: BGE-M3 ---
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={'normalize_embeddings': True},
        )
        # --- 2. Подключение к Qdrant ---
        client = QdrantClient(url=qdrant_url)
        self.vectorstore = Qdrant(
            embeddings=self.embedding_model,
            client=client,
            collection_name=self.collection_name
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # --- 3. Генеративная модель: Qwen-1.8B ---
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            trust_remote_code=True
        )

        # Пайплайн генерации
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.1,
            trust_remote_code=True
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # --- 4. Промт ---
        if prompt_template is None:
            prompt_template = """System: Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги. .
Если не знаешь ответа, скажи: "Я не знаю".



Контекст:
{context}

Вопрос: {question}
Ответ:"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # --- 5. Цепочка RAG ---
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def ask(self, question: str):
        """
        Задать вопрос системе.
        Возвращает словарь с ответом и источниками.
        """
        result = self.qa_chain.invoke({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }

    def get_relevant_documents(self, query: str, k: int = 3):
        """
        Получить только релевантные документы (без генерации).
        Полезно для отладки.
        """
        return self.retriever.get_relevant_documents(query, k=k)