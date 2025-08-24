from langchain_community.vectorstores import Qdrant
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from qdrant_client import QdrantClient
from typing import List, Dict, Any
import torch

class RAGSystem:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model_name: str = "BAAI/bge-m3",
        llm_model_name: str = "Qwen/Qwen1.5-1.8B-Chat",
        prompt_template: str = None,
        device: str = "cuda"
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.device = device

        # --- 1. Эмбеддинги ---
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={'normalize_embeddings': True},
        )
        
        # --- 2. Подключение к Qdrant ---
        client = QdrantClient(url=qdrant_url)
        self.vectorstore = Qdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # --- 3. Генеративная модель ---
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        print("tokenizer loaded")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        # Ваш вопрос
        prompt = "how are you working?"

        # Подготовка входа
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Генерация ответа
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Декодирование и вывод результата
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Убираем промпт из ответа, если нужно
        answer = response[len(prompt):].strip()

        print("Ответ модели:", answer)
      
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
            pad_token_id=self.tokenizer.eos_token_id,  # Важно!
            trust_remote_code=True
        )

        print("pipe created")
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # --- 4. Промт ---
        if prompt_template is None:
            prompt_template = """System: Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги.
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

    def ask(self, question: str) -> Dict[str, Any]:
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

    def get_relevant_documents(self, query: str, k: int = 3) -> List:
        """
        Получить только релевантные документы (без генерации).
        """
        # Исправленный вызов
        return self.retriever.invoke(query, config={"k": k})