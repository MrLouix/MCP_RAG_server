"""Tests for hashing utilities."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from mcp_rag.utils.hashing import sha256_bytes, sha256_file, short_doc_id


def test_sha256_bytes():
    data = b"hello world"
    h = sha256_bytes(data)
    assert len(h) == 64
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_sha256_file():
    with NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"test content")
        tmp.flush()
        path = Path(tmp.name)
    h = sha256_file(path)
    assert len(h) == 64
    # Cleanup
    path.unlink()


def test_short_doc_id():
    full = "abcdef1234567890" * 4
    short = short_doc_id(full, prefix_len=8)
    assert short == "abcdef12"
    assert len(short) == 8
