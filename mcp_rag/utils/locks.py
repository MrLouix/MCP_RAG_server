"""Per-doc_id asyncio locks to prevent concurrent ingestion races."""

import asyncio
from typing import Any


class DocLockRegistry:
    """Registry of asyncio.Lock keyed by doc_id."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def acquire(self, doc_id: str) -> asyncio.Lock:
        """Get or create a lock for the given doc_id."""
        if doc_id not in self._locks:
            self._locks[doc_id] = asyncio.Lock()
        return self._locks[doc_id]

    async def __aenter__(self) -> "DocLockRegistry":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
