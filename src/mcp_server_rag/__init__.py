import click
from pathlib import Path
import logging
import sys
from .server import serve

@click.command()
@click.option("--docs-dir", "-d", type=Path, help="문서 디렉토리 경로")
@click.option("--vector-store-dir", "-v", type=Path, help="벡터 저장소 디렉토리 경로")
@click.option("--verbose", count=True, help="로깅 상세 수준 증가")
def main(docs_dir: Path | None = None, vector_store_dir: Path | None = None, verbose: bool = False) -> None:
    """MCP RAG Server - 문서 검색 및 처리를 위한 MCP 서버"""
    import asyncio

    # 로깅 수준 설정
    logging_level = logging.WARN
    if verbose == 1:
        logging_level = logging.INFO
    elif verbose >= 2:
        logging_level = logging.DEBUG

    # 로깅 설정
    logging.basicConfig(
        level=logging_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    # 서버 실행
    asyncio.run(serve(docs_dir, vector_store_dir))

if __name__ == "__main__":
    main()
