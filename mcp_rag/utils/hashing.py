"""Content hashing utilities for deduplication and doc_id generation."""

import hashlib
from pathlib import Path


CHUNK_SIZE = 8192


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def short_doc_id(full_hash: str, prefix_len: int = 8) -> str:
    """Return a short doc_id prefix from a full SHA-256 hash."""
    return full_hash[:prefix_len]
