"""CLI for manual tagging/re-tagging outside the MCP server process."""

import argparse
import asyncio
import sys
from pathlib import Path


async def tag(file_path: Path, force: bool = False) -> int:
    print(f"Tagging: {file_path} (force={force})")
    # TODO: wire to tagging engine
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag a document using the local LLM")
    parser.add_argument("file", type=Path, help="File to tag")
    parser.add_argument("-f", "--force", action="store_true", help="Force retag")
    args = parser.parse_args()
    sys.exit(asyncio.run(tag(args.file, args.force)))


if __name__ == "__main__":
    main()
