# FastAPI 없이 백엔드만 활용하는 독립적 인덱싱 스크립트

from main import create_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

if __name__ == "__main__":
    print("문서 인덱싱 시작...")
    persist_directory = Path("vector_store")
    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"trust_remote_code": True}
    )
    create_vector_store(embeddings, persist_directory)
    print("✅ 인덱싱 완료")
