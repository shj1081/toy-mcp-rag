# index_document.py
"""
문서 인덱싱 클라이언트 스크립트
이 스크립트는 새 문서를 RAG 서버에 인덱싱하는 클라이언트 역할을 합니다.
"""
import sys
import json
import asyncio
import argparse
from pathlib import Path

from mcp.client import Client
from mcp.types import CallToolInput

async def index_document(doc_path: str):
    """
    지정된 문서를 RAG 서버에 인덱싱합니다.
    
    Args:
        doc_path: 인덱싱할 문서 파일 경로
    """
    file_path = Path(doc_path)
    if not file_path.exists() or not file_path.is_file():
        print(f"오류: 파일을 찾을 수 없습니다: {doc_path}")
        return
        
    print(f"인덱싱할 문서: {file_path.name}")
    print("RAG 서버에 연결 중...")
    
    async with Client.connect_stdio('mcp-rag') as client:
        tool_input = CallToolInput(
            name="rag_index",
            arguments={
                "doc_path": str(file_path.absolute())
            }
        )
        
        print("인덱싱 요청 중...")
        response = await client.call_tool(tool_input)
        
        print("\n=== 인덱싱 결과 ===")
        for content in response.content:
            print(content.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 인덱싱 도구")
    parser.add_argument("doc_path", help="인덱싱할 문서 파일 경로")
    args = parser.parse_args()
    
    asyncio.run(index_document(args.doc_path))
