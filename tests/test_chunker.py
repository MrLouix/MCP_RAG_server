"""Tests for the document chunker."""

import pytest

# Skip if langchain is not yet installed (build in progress)
pytest.importorskip("langchain")

from mcp_rag.chunker import Chunker


def test_simple_split():
    text = "\n\n".join(["Paragraph number " + str(i) for i in range(100)])
    c = Chunker(chunk_size=200, chunk_overlap=20)
    chunks = c.split(text)
    assert len(chunks) > 1
    assert all(len(ch) > 0 for ch in chunks)


def test_max_chunks_limit():
    text = "word " * 50000
    c = Chunker(chunk_size=10, chunk_overlap=0)
    chunks = c.split(text, max_chunks=5)
    assert len(chunks) == 5
