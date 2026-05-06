"""CLI for batch ingestion outside the MCP server process."""

import argparse
import asyncio
import sys
from pathlib import Path

from mcp_rag.config import load_config


async def ingest(path: Path, recursive: bool = True, config_path: str | None = None) -> int:
    cfg = load_config(config_path)
    print(f"Ingestion: {path}")
    print(f"Config: {cfg}")
    # TODO: wire to ingest pipeline
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument("path", type=Path, help="Directory or file to ingest")
    parser.add_argument("-r", "--recursive", action="store_true", default=True, help="Recursive")
    parser.add_argument("-c", "--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()
    sys.exit(asyncio.run(ingest(args.path, args.recursive, args.config)))


if __name__ == "__main__":
    main()
