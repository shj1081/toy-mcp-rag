# test_query.py
"""
RAG 문서 검색 클라이언트 테스트 스크립트
이 스크립트는 독립적으로 RAG 서버에 쿼리를 수행하는 클라이언트 역할을 합니다.
"""
import sys
import json
import asyncio
from pathlib import Path

from mcp.client import Client
from mcp.types import CallToolInput

async def test_query():
    """RAG 서버에 쿼리를 전송하고 결과를 출력합니다."""
    query = "네트워크 계층에서 패킷이 전달되는 방식"
    
    print(f"검색 쿼리: {query}")
    print("RAG 서버에 연결 중...")
    
    async with Client.connect_stdio('mcp-rag') as client:
        tool_input = CallToolInput(
            name="rag_search",
            arguments={
                "query": query,
                "limit": 10,
                "threshold": 0.1
            }
        )
        
        print("쿼리 전송 중...")
        response = await client.call_tool(tool_input)
        
        print("\n=== 검색 결과 ===")
        for content in response.content:
            print(content.text)

if __name__ == "__main__":
    asyncio.run(test_query())
