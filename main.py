"""
MCP(Model Context Protocol) RAG 문서 서버

이 모듈은 텍스트, PDF, 마크다운 형식의 문서를 로드하고 벡터화하여
효율적인 의미 기반 검색(Semantic Search)을 제공하는 서버를 구현합니다.

주요 기능:
- 다양한 형식의 문서 자동 로딩 및 처리
- 문서 변경 감지 및 벡터 저장소 자동 업데이트
- 다국어 지원 임베딩 모델을 통한 의미 기반 검색
- RESTful API를 통한 검색 서비스 제공

사용법:
  uv run main.py
"""

import os
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 로깅 설정: 파일 및 콘솔 출력 모두 지원
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_server.log"),  # 로그 파일에 기록
        logging.StreamHandler()                 # 콘솔에도 출력
    ]
)

# 문서 처리 관련 라이브러리
from langchain_community.document_loaders import (
    TextLoader,          # 일반 텍스트 파일 로더
    DirectoryLoader,     # 디렉토리 내 파일 일괄 로딩
    PyPDFLoader,         # 일반 PDF 파일 로더 (작은 파일용)
    UnstructuredMarkdownLoader,  # 마크다운 파일 로더
    UnstructuredPDFLoader,       # 고급 PDF 로더 (큰 파일용)
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 문서를 작은 청크로 분할
from langchain_chroma import Chroma  # 벡터 데이터베이스 (ChromaDB)
from langchain_huggingface import HuggingFaceEmbeddings  # 허깅페이스 임베딩 모델 연동

# MCP 서버 인스턴스 초기화
mcp = FastMCP("MCP(model context protocol) RAG Documents Server")

# 문서 파일이 저장된 기본 경로 설정
RESOURCES_PATH = Path("docs")

@mcp.tool()
def search_rag_docs(query: str) -> str:
    """
    모든 문서에서 특정 쿼리를 벡터 검색하는 도구 함수입니다.
    
    Args:
        query: 검색할 텍스트 쿼리
        
    Returns:
        str: 마크다운 형식의 검색 결과 또는 오류 메시지
    """
    try:
        # 리소스 디렉토리 유효성 검증
        if not RESOURCES_PATH.exists() or not RESOURCES_PATH.is_dir():
            return "리소스 디렉토리를 찾을 수 없거나 디렉토리가 아닙니다"
        
        # 벡터 저장소 영구 디렉토리 설정
        persist_directory = Path("vector_store")
        persist_directory.mkdir(exist_ok=True)
        
        # 문서 변경 감지용 해시 파일 설정
        hash_file = persist_directory / "docs_hash.txt"
        
        # 현재 문서 컬렉션의 해시값 계산
        current_hash = calculate_docs_hash(RESOURCES_PATH)
        
        # 다국어 지원 임베딩 모델 설정
        # GTE-Multilingual 모델은 100+ 언어 지원 (한국어 포함)
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={'trust_remote_code': True}
        )
        
        # 벡터 저장소 초기화 또는 로딩 로직
        vector_store = initialize_vector_store(
            hash_file, current_hash, persist_directory, embeddings
        )
            
        if vector_store is None:
            return "벡터 저장소를 초기화할 수 없습니다."
            
        # 유사도 검색 수행 (상위 10개)
        results = vector_store.similarity_search_with_score(query, k=10)

        # 의미적 유사도가 임계값(0.1) 이상인 결과만 필터링
        relevant_results = [result for result in results if float(result[1]) >= 0.1]
        
        # 검색 결과 없음 처리
        if not relevant_results:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        
        # 결과를 마크다운 형식으로 포맷팅
        return format_search_results(relevant_results)
        
    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return f"벡터 검색 중 오류 발생: {str(e)}"

def calculate_docs_hash(resource_path: Path) -> str:
    """
    문서 디렉토리의 현재 상태에 대한 해시값을 계산합니다.
    파일명, 수정 시간, 크기를 조합하여 고유한 해시값 생성.
    
    Args:
        resource_path: 문서 디렉토리 경로
        
    Returns:
        str: 문서 컬렉션의 MD5 해시값
    """
    hash_input = ""
    for file_path in resource_path.glob("**/*.*"):
        if file_path.is_file():
            file_stat = os.stat(file_path)
            # 파일명, 수정시간, 크기를 조합
            file_info = f"{file_path.name}:{file_stat.st_mtime}:{file_stat.st_size}"
            hash_input += file_info
    
    # MD5 해시 생성
    return hashlib.md5(hash_input.encode()).hexdigest()

def initialize_vector_store(hash_file: Path, current_hash: str, 
                           persist_directory: Path, embeddings) -> Optional[Chroma]:
    """
    문서 변경 감지 및 벡터 저장소 초기화 함수
    
    Args:
        hash_file: 해시값 저장 파일 경로
        current_hash: 현재 문서의 해시값
        persist_directory: 벡터 저장소 경로
        embeddings: 임베딩 모델
        
    Returns:
        Optional[Chroma]: 초기화된 벡터 저장소 또는 None (오류 시)
    """
    try:
        if hash_file.exists():
            # 이전 해시 파일 읽기
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()
            
            # 해시값 비교로 문서 변경 감지
            if current_hash == stored_hash and os.path.exists(persist_directory / "chroma.sqlite3"):
                logging.info("문서 변경 없음. 기존 벡터 저장소 로드")
                # 문서 변경 없음, 기존 벡터 저장소 사용
                return Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=embeddings
                )
            else:
                logging.info("문서 변경 감지됨. 벡터 저장소 재생성")
                # 문서 변경 있음, 벡터 저장소 재생성
                vector_store = create_vector_store(embeddings, persist_directory)
                # 새 해시값 저장
                with open(hash_file, "w") as f:
                    f.write(current_hash)
                return vector_store
        else:
            logging.info("초기 실행. 벡터 저장소 생성")
            # 처음 실행 시 벡터 저장소 생성
            vector_store = create_vector_store(embeddings, persist_directory)
            # 해시값 저장
            with open(hash_file, "w") as f:
                f.write(current_hash)
            return vector_store
    except Exception as e:
        logging.error(f"벡터 저장소 초기화 오류: {str(e)}")
        return None

def format_search_results(results) -> str:
    """
    검색 결과를 마크다운 형식으로 포맷팅
    
    Args:
        results: 유사도 검색 결과 리스트 [(document, score), ...]
        
    Returns:
        str: 마크다운 형식의 결과 문자열
    """
    formatted_results = ["# 검색 결과\n"]
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "알 수 없는 소스")
        source_name = Path(source).name
        # 결과 헤더 (번호, 파일명, 유사도 점수)
        formatted_results.append(f"## 결과 {i} - {source_name} (유사도: {score:.4f})\n")
        # 문서 내용 코드 블록
        formatted_results.append(f"```\n{doc.page_content}\n```\n")
    
    return "\n".join(formatted_results)

def create_vector_store(embeddings, persist_directory):
    """
    다양한 형식의 문서를 로드하고 벡터 저장소를 생성하는 함수
    
    지원 파일 형식:
    - 텍스트 파일 (.txt)
    - PDF 파일 (.pdf) - 크기에 따라 다른 로더 사용
    - 마크다운 파일 (.md)
    
    Args:
        embeddings: 사용할 임베딩 모델
        persist_directory: 벡터 저장소 저장 경로
        
    Returns:
        Chroma: 생성된 벡터 저장소 객체
    """
    # 문서 컬렉션 리스트
    all_documents = []

    # 1. 텍스트 파일 로딩
    try:
        text_loader = DirectoryLoader(
            str(RESOURCES_PATH),
            glob="**/*.txt",  # 모든 하위 폴더 포함 .txt 파일
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}  # UTF-8 인코딩 명시
        )
        text_documents = text_loader.load()
        all_documents.extend(text_documents)
        logging.info(f"텍스트 파일 {len(text_documents)}개 로드 완료")
    except Exception as e:
        logging.error(f"텍스트 파일 로드 중 오류: {str(e)}")

    # 2. PDF 파일 로딩 (크기에 따른 최적화)
    pdf_files = list(RESOURCES_PATH.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        try:
            # 파일 크기 확인 (MB 단위)
            file_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            
            if file_size_mb > 20:  # 20MB 이상 대용량 PDF
                # 대용량 PDF는 고급 파서로 처리 (더 정확하지만 느림)
                loader = UnstructuredPDFLoader(
                    str(pdf_file),
                    mode="elements",  # 요소별 추출 (텍스트, 표, 이미지 등)
                    strategy="fast"   # 빠른 처리 전략
                )
                logging.info(f"대용량 PDF 로드 중 (Unstructured): {pdf_file.name} ({file_size_mb:.2f}MB)")
            else:
                # 일반 PDF는 기본 파서로 처리 (빠르지만 단순)
                loader = PyPDFLoader(str(pdf_file))
                logging.info(f"PDF 로드 중 (PyPDF): {pdf_file.name} ({file_size_mb:.2f}MB)")
                
            pdf_documents = loader.load()
            all_documents.extend(pdf_documents)
            logging.info(f"PDF 파일 로드 완료: {pdf_file.name} - {len(pdf_documents)}개 페이지/요소")
        except Exception as e:
            logging.error(f"PDF 파일 {pdf_file.name} 로드 중 오류: {str(e)}")

    # 3. 마크다운 파일 로딩
    md_files = list(RESOURCES_PATH.glob("**/*.md"))
    for md_file in md_files:
        try:
            loader = UnstructuredMarkdownLoader(str(md_file))
            md_documents = loader.load()
            all_documents.extend(md_documents)
            logging.info(f"마크다운 파일 로드 완료: {md_file.name}")
        except Exception as e:
            logging.error(f"마크다운 파일 {md_file.name} 로드 중 오류: {str(e)}")

    # 로드된 문서 확인
    if not all_documents:
        raise ValueError("로드할 문서를 찾을 수 없습니다. docs 디렉토리에 .txt, .pdf, .md 파일이 있는지 확인하세요.")
    
    logging.info(f"총 {len(all_documents)}개 문서 로드 완료")
    
    # 문서 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,     # 각 청크의 최대 문자 수
        chunk_overlap=400,   # 청크 간 중복 영역 (문맥 연결성 유지)
        length_function=len,
        # 분할 우선순위: 빈 줄, 줄바꿈, 공백, 문자 단위
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 문서 분할 실행
    chunks = text_splitter.split_documents(all_documents)
    logging.info(f"문서 분할 완료: {len(chunks)}개 청크 생성")
    
    # 벡터 저장소 생성 및 저장
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    
    logging.info(f"벡터 저장소 생성 완료: {persist_directory}")
    return vector_store

# 메인 실행 지점
if __name__ == "__main__":
    logging.info("MCP RAG 문서 서버 시작")
    mcp.run()