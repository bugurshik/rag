# rag_qwen_qdrant.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Dict
from retrieval_module import DataRetrieval


device = "cuda" if torch.cuda.is_available() else "cpu"
system_promt = "Ты — помощник, использующий контекст для ответов."

class RAGWithQdrant:
    def __init__(
            self, 
            llm_model:str,
            retrieval:DataRetrieval,
        ):
        self.retrieval = retrieval

        # 3. Загрузка Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Генерация ответа через Qwen"""
        context_text = "\n".join([f"- {doc}" for doc in context])
        prompt = f"""
        Отвечай точно на основе контекста. Если не знаешь — скажи: "Я не знаю."

        Контекст:
        {context_text}

        Вопрос: {query}
        Ответ:
        """.strip()

        messages = [
            {"role": "system", "content": system_promt},
            {"role": "user", "content": prompt}
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            return f"Ошибка генерации: {str(e)}"

    def ask(self, query: str, k: int = 3) -> Dict[str, any]:
        retrieved = self.retrieval.retrieve(query, k=k)
        answer = self.generate_answer(query, retrieved)
        return {
            "query A": query,
            "retrieved": retrieved,
            "answer": answer
        }