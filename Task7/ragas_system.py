import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_huggingface import HuggingFaceEndpoint
from utils import read_jsonl
from langchain_community.embeddings import HuggingFaceEmbeddings

def ragas_evaluate(logs_file:str):
    logs = read_jsonl(logs_file)

    def prepare_ragas_data(logs):
        ragas_data = []
        
        for log in logs:
            ragas_data.append({
                "question": log["question"],
                "answer": log["answer"],
                "contexts": log["sources"]
            })
        
        return ragas_data

    # Установить датасет RAGAS
    ragas_formatted_data = prepare_ragas_data(logs)

    print("------contexts-------")
    print(ragas_formatted_data)
    print("------contexts-------")

    ds = Dataset.from_list(ragas_formatted_data)
    from openai import OpenAI
    client = OpenAI(
        api_key="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",  # может быть любым, если сервер не требует ключ
        base_url="https://models.inference.ai.azure.com"
    )
  
    # 4. Получить метрики
    langchain_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceTB/SmolLM3-3B",
        provider="hf-inference",
        temperature=0,
        do_sample=False,
        task='text-generation',
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={"device": 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )

    custom_embeddings = LangchainEmbeddingsWrapper(embeddings=embedding_model)
    # custom_llm = LangchainLLMWrapper(langchain_llm=langchain_llm)
    custom_llm = LangchainLLMWrapper(langchain_llm=client)
    report = evaluate(ds, llm=custom_llm, embeddings=custom_embeddings, metrics=[faithfulness, answer_relevancy])
    print(report)
    return report