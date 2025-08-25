import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_huggingface import HuggingFaceEndpoint
from utils import read_jsonl
from langchain_community.embeddings import HuggingFaceEmbeddings

def ragas_evaluate(logs_file:str):
    logs = read_jsonl(logs_file)

    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={"device": 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )

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

    # 4. Получить метрики
    langchain_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Devstral-Small-2505",
        provider="nebius",
        max_new_tokens=256,
        temperature=0,
        do_sample=False,
        task='text-generation',
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    custom_embeddings = LangchainEmbeddingsWrapper(embeddings=embedding_model)
    custom_llm = LangchainLLMWrapper(langchain_llm=langchain_llm)

    report = evaluate(ds, llm=custom_llm, embeddings=custom_embeddings, metrics=[faithfulness, answer_relevancy])
    print(report)
    return report