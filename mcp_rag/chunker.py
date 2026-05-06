"""Chunking strategy for document segmentation.

Wrapper around LangChain's RecursiveCharacterTextSplitter with consistent defaults.
"""

import logging
from typing import Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class Chunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str, max_chunks: int = 1500) -> list[str]:
        """Split text into chunks. Returns at most max_chunks."""
        chunks = self._splitter.split_text(text)
        if len(chunks) > max_chunks:
            logger.warning(
                "max_chunks_exceeded",
                extra={
                    "chunks": len(chunks),
                    "max_chunks": max_chunks,
                    "truncated": True,
                },
            )
            return chunks[:max_chunks]
        return chunks

    def split_texts(self, texts: Sequence[str]) -> list[str]:
        """Split multiple texts and return all chunks as a flat list."""
        out: list[str] = []
        for t in texts:
            out.extend(self.split(t))
        return out
