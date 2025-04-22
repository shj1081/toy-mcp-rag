"""
MCP(Model Context Protocol) RAG 문서 서버

이 모듈은 텍스트, PDF, 마크다운 형식의 문서를 로드하고 벡터화하여
효율적인 의미 기반 검색(Semantic Search)을 제공하는 서버를 구현합니다.

주요 기능:
- 다양한 형식의 문서 자동 로딩 및 처리
- 파일 단위 변경 감지 및 증분 업데이트
- 다국어 지원 임베딩 모델을 통한 의미 기반 검색
- RESTful API를 통한 검색 서비스 제공

개선사항:
- 파일 단위 해시 관리로 변경된 파일만 처리
- 파일 삭제 시 해당 파일 관련 벡터만 삭제
- 리소스 효율성 및 확장성 강화
"""

import os
import hashlib
import json
import asyncio
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

# 지원하는 파일 확장자 정의
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.md']

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
        
        # 다국어 지원 임베딩 모델 설정
        # GTE-Multilingual 모델은 100+ 언어 지원 (한국어 포함)
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={'trust_remote_code': True}
        )
        
        # 파일별 변경 감지 및 선택적 업데이트
        vector_store = manage_vector_store(embeddings, persist_directory)
            
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
        logging.error(f"검색 처리 중 오류: {str(e)}")
        return f"벡터 검색 중 오류 발생: {str(e)}"

def calculate_file_hash(file_path: Path) -> str:
    """
    단일 파일의 해시값을 계산합니다.
    파일명, 수정 시간, 크기를 조합하여 해시값 생성.
    
    Args:
        file_path: 파일 경로
        
    Returns:
        str: 파일의 MD5 해시값
    """
    file_stat = os.stat(file_path)
    file_info = f"{file_path}:{file_stat.st_mtime}:{file_stat.st_size}"
    return hashlib.md5(file_info.encode()).hexdigest()

def manage_vector_store(embeddings, persist_directory: Path):
    """
    파일별 변경 사항을 추적하고 필요한 파일만 업데이트하는 함수
    
    Args:
        embeddings: 임베딩 모델
        persist_directory: 벡터 저장소 저장 경로
        
    Returns:
        Chroma: 벡터 저장소 객체
    """
    # 파일별 해시값을 저장할 JSON 파일 경로
    hash_file = persist_directory / "file_hashes.json"
    stored_hashes = {}
    
    # 기존 해시값 로드
    if hash_file.exists():
        try:
            with open(hash_file, "r") as f:
                stored_hashes = json.load(f)
        except json.JSONDecodeError:
            logging.warning("손상된 해시 파일 감지. 파일 해시 재계산을 시작합니다.")
    
    # 현재 파일 목록과 해시값 계산
    current_files = {}
    all_file_paths = []
    
    for file_path in RESOURCES_PATH.glob("**/*.*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            path_str = str(file_path)
            file_hash = calculate_file_hash(file_path)
            current_files[path_str] = file_hash
            all_file_paths.append(file_path)
    
    # 문서 파일이 하나도 없을 경우 처리
    if not all_file_paths:
        logging.warning("로드할 문서 파일이 없습니다. docs 디렉토리를 확인하세요.")
        return None
    
    # 벡터 저장소 초기화 또는 로드
    vector_store = None
    db_file = persist_directory / "chroma.sqlite3"
    
    if db_file.exists():
        # 기존 벡터 저장소가 있으면 로드
        logging.info("기존 벡터 저장소 로드 중...")
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings
        )
        
        # 변경 파일 확인 및 처리
        if stored_hashes:  # 이전 해시 정보가 있는 경우
            # 새로 추가된 파일 또는 수정된 파일 확인
            new_or_modified = []
            for file_path, file_hash in current_files.items():
                if file_path not in stored_hashes or stored_hashes[file_path] != file_hash:
                    new_or_modified.append(Path(file_path))
                    logging.info(f"변경 감지: {Path(file_path).name}")
            
            # 삭제된 파일 처리
            deleted_files = [f for f in stored_hashes if f not in current_files]
            for file_path in deleted_files:
                logging.info(f"삭제 감지: {Path(file_path).name}")
                # 해당 파일 관련 벡터만 삭제
                vector_store.delete(where={"source": file_path})
            
            # 새 파일/수정 파일만 추가 처리
            if new_or_modified:
                logging.info(f"{len(new_or_modified)}개 파일에 대한 증분 업데이트 수행")
                add_files_to_vector_store(new_or_modified, vector_store, embeddings)
            else:
                logging.info("변경된 파일 없음. 벡터 저장소 유지")
        else:
            # 이전 해시 정보가 없지만 DB는 있는 비정상 상태
            logging.warning("벡터 저장소는 있으나 해시 정보가 없습니다. 전체 재구성을 시작합니다.")
            vector_store = create_full_vector_store(all_file_paths, embeddings, persist_directory)
    else:
        # 초기 실행 시 전체 벡터 저장소 생성
        logging.info("벡터 저장소 최초 생성 중...")
        vector_store = create_full_vector_store(all_file_paths, embeddings, persist_directory)
    
    # 새 해시값 저장
    with open(hash_file, "w") as f:
        json.dump(current_files, f)
    
    return vector_store

def add_files_to_vector_store(file_paths, vector_store, embeddings):
    """
    지정된 파일들만 처리하여 기존 벡터 저장소에 추가
    
    Args:
        file_paths: 처리할 파일 경로 목록
        vector_store: 기존 벡터 저장소
        embeddings: 임베딩 모델
    """
    documents = []
    
    for file_path in file_paths:
        try:
            # 파일 형식에 따라 적절한 로더 선택
            if file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_path.suffix.lower() == '.pdf':
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > 20:
                    loader = UnstructuredPDFLoader(str(file_path), mode="elements", strategy="fast")
                else:
                    loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                continue
                
            # 문서 로드
            file_docs = loader.load()
            documents.extend(file_docs)
            logging.info(f"파일 로드 완료: {file_path.name}")
            
        except Exception as e:
            logging.error(f"파일 {file_path.name} 처리 중 오류: {str(e)}")
    
    if documents:
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # 기존 벡터 저장소에 추가
        vector_store.add_documents(chunks)
        logging.info(f"{len(chunks)}개 청크가 벡터 저장소에 추가됨")
    else:
        logging.warning("추가할 문서가 없습니다.")

def create_full_vector_store(file_paths, embeddings, persist_directory):
    """
    전체 벡터 저장소를 새로 생성하는 함수
    
    Args:
        file_paths: 처리할 파일 경로 목록
        embeddings: 임베딩 모델
        persist_directory: 벡터 저장소 저장 경로
        
    Returns:
        Chroma: 생성된 벡터 저장소 객체
    """
    # 문서 컬렉션 리스트
    all_documents = []
    
    # 파일 형식별 로드 시도
    for file_path in file_paths:
        try:
            # 파일 형식에 따라 적절한 로더 선택
            if file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                all_documents.extend(docs)
                logging.info(f"텍스트 파일 로드 완료: {file_path.name}")
            
            elif file_path.suffix.lower() == '.pdf':
                # 파일 크기 확인 (MB 단위)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                if file_size_mb > 20:  # 20MB 이상 대용량 PDF
                    # 대용량 PDF는 고급 파서로 처리
                    loader = UnstructuredPDFLoader(
                        str(file_path),
                        mode="elements",
                        strategy="fast"
                    )
                    logging.info(f"대용량 PDF 로드 중: {file_path.name} ({file_size_mb:.2f}MB)")
                else:
                    # 일반 PDF는 기본 파서로 처리
                    loader = PyPDFLoader(str(file_path))
                    logging.info(f"PDF 로드 중: {file_path.name} ({file_size_mb:.2f}MB)")
                    
                pdf_documents = loader.load()
                all_documents.extend(pdf_documents)
                logging.info(f"PDF 파일 로드 완료: {file_path.name} - {len(pdf_documents)}개 페이지/요소")
            
            elif file_path.suffix.lower() == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
                md_documents = loader.load()
                all_documents.extend(md_documents)
                logging.info(f"마크다운 파일 로드 완료: {file_path.name}")
            
        except Exception as e:
            logging.error(f"파일 {file_path.name} 로드 중 오류: {str(e)}")

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

# 메인 실행 지점
if __name__ == "__main__":
    logging.info("MCP RAG 문서 서버 시작")
    asyncio.run(mcp.run())
