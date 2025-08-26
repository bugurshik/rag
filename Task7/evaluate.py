from dotenv import load_dotenv
import os
from rag_system import RAGSystem
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import uuid
import datetime
from utils import save_as_jsonl, read_as_list
from ragas_system import ragas_evaluate
import json

load_dotenv() 
sys.stdout.reconfigure(encoding="utf-8")

LLM_MODEL_NAME = "Qwen/Qwen2.5-3B"
EMBEDDING_MODEL = "BAAI/bge-m3"

QDRANT_URL= "http://localhost:6333"
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

LOG_FILE_PATH = 'Task7/copy.jsonl'
GOLDEN_QUESTION_FILE_PATH= 'Task7/golden_questions.txt'

# --- 1. Эмбеддинги ---
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
)

# --- 2. Подключение к Qdrant ---
vectorstore = Qdrant(
    client=QdrantClient(url=QDRANT_URL),
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

rag = RAGSystem(
    retriever= vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm_model_name=LLM_MODEL_NAME,
)

with open('Task7/golden_questions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print(data)
eval_questions = read_as_list(GOLDEN_QUESTION_FILE_PATH)
print(eval_questions)

logs = []
for golden_question in data:
    start_time = datetime.datetime.now()
    result = rag.ask(golden_question['question'])
    print(result)
    end_time = datetime.datetime.now()
    log_entry = {
        "id": str(uuid.uuid4()),
        "question": golden_question['question'],
        "timestamp": start_time.isoformat(),
        "has_context": len(result["sources"]) > 1,
        "answer": result["answer"],
        "answer_len": len(result["answer"]),
        "reference" : golden_question['answer'],
        "sources": [doc['content'] for doc in result["sources"]],
        "sources_metadata": [doc for doc in result["sources"]],
    }

    logs.append(log_entry)

save_as_jsonl(LOG_FILE_PATH, logs)

ragas_evaluate(LOG_FILE_PATH)