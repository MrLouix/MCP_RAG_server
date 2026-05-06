"""Storage layer — ChromaDB abstraction + SQLite inverted path index.

Handles:
- Collection lifecycle (multi-workspace, schema_version, embedding_model metadata)
- Chunk CRUD with bulk insert
- Semantic search with tag filtering
- Inverted path index (SQLite) for filesystem sync / watcher
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
from chromadb.config import Settings

logger = logging.getLogger(__name__)

DOC_META_KEYS = {"doc_id", "source_path", "source_name", "file_type", "content_hash", "schema_version"}


class Storage:
    """ChromaDB persistent storage with per-workspace collections."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.index_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_collection(self, workspace: str = "default", embedding_model: str = "") -> Any:
        """Get or create a collection for the given workspace."""
        if workspace in self._collections:
            return self._collections[workspace]

        coll_name = _sanitize_name(workspace)
        try:
            coll = self.client.get_collection(name=coll_name)
            # Validate embedding model consistency
            meta = coll.metadata or {}
            stored_model = meta.get("embedding_model", "")
            if stored_model and embedding_model and stored_model != embedding_model:
                raise ValueError(
                    f"Embedding model mismatch: collection uses '{stored_model}', "
                    f"but config requests '{embedding_model}'. Run reindex_all."
                )
        except Exception:
            coll = self.client.create_collection(
                name=coll_name,
                metadata={
                    "embedding_model": embedding_model,
                    "schema_version": "1.0",
                },
            )
        self._collections[workspace] = coll
        return coll

    def list_collections(self) -> list[str]:
        """Return all workspace collection names."""
        raw = self.client.list_collections()
        # ChromaDB API varies; normalize
        names = []
        for c in raw:
            if isinstance(c, str):
                names.append(c)
            else:
                names.append(c.name)
        return names

    def delete_collection(self, workspace: str = "default") -> None:
        """Drop a workspace collection entirely."""
        coll_name = _sanitize_name(workspace)
        try:
            self.client.delete_collection(name=coll_name)
        except Exception:
            pass
        self._collections.pop(workspace, None)

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        doc_id: str,
        chunks: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
        workspace: str = "default",
        embedding_model: str = "",
    ) -> int:
        """Bulk insert chunks into the collection. Returns number inserted."""
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            raise ValueError("chunks / embeddings / metadatas length mismatch")

        coll = self.get_collection(workspace, embedding_model)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        # Ensure embeddings are list[list[float]]
        vectors = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

        coll.add(
            ids=ids,
            documents=list(chunks),
            embeddings=vectors,
            metadatas=list(metadatas),
        )
        return len(chunks)

    def delete_doc_chunks(self, doc_id: str, workspace: str = "default", embedding_model: str = "") -> int:
        """Delete all chunks belonging to a doc_id. Returns count."""
        coll = self.get_collection(workspace, embedding_model)
        try:
            coll.delete(where={"doc_id": doc_id})
        except Exception:
            # Fallback: list and delete by IDs
            pass
        return -1  # ChromaDB doesn't always return count from delete

    def update_doc_metadata(
        self,
        doc_id: str,
        updates: dict[str, Any],
        workspace: str = "default",
        embedding_model: str = "",
    ) -> int:
        """Update metadata on all chunks of a doc_id."""
        coll = self.get_collection(workspace, embedding_model)
        try:
            coll.update(where={"doc_id": doc_id}, metadatas=updates)
        except Exception:
            # Older ChromaDB versions may not support update with where+
            pass
        return 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray | Sequence[float],
        top_k: int = 5,
        tags: list[str] | None = None,
        tags_mode: str = "all",
        filters: dict[str, Any] | None = None,
        workspace: str = "default",
        embedding_model: str = "",
    ) -> list[dict[str, Any]]:
        """Semantic search with optional tag and property filters."""
        coll = self.get_collection(workspace, embedding_model)
        where_clause = _build_where_clause(tags, tags_mode, filters)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            kwargs["where"] = where_clause

        results = coll.query(**kwargs)
        return _normalize_results(results)

    def get_document_chunks(
        self,
        doc_id: str,
        workspace: str = "default",
        embedding_model: str = "",
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks for a specific doc_id."""
        coll = self.get_collection(workspace, embedding_model)
        try:
            results = coll.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            results = {}
        return _normalize_get_results(results)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count_chunks(self, workspace: str = "default", embedding_model: str = "") -> int:
        """Total chunks in a workspace."""
        coll = self.get_collection(workspace, embedding_model)
        return coll.count()

    def list_doc_ids(self, workspace: str = "default", embedding_model: str = "") -> set[str]:
        """Return unique doc_ids present in the workspace."""
        coll = self.get_collection(workspace, embedding_model)
        try:
            all_meta = coll.get(include=["metadatas"])
            metas = all_meta.get("metadatas", [])
        except Exception:
            metas = []
        return {m["doc_id"] for m in metas if m and "doc_id" in m}

    # ------------------------------------------------------------------
    # Inverted path index (SQLite) — for watcher sync
    # ------------------------------------------------------------------

    def _ensure_path_index(self) -> sqlite3.Connection:
        """Lazy-connect to the SQLite path index."""
        db_path = self.index_path / "path_index.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS path_index "
            "(abs_path TEXT PRIMARY KEY, doc_id TEXT, workspace TEXT, modified_at REAL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_id ON path_index(doc_id)"
        )
        return conn

    def upsert_path_index(
        self,
        abs_path: str,
        doc_id: str,
        workspace: str = "default",
        modified_at: float | None = None,
    ) -> None:
        """Register (or update) a path ↔ doc_id mapping."""
        conn = self._ensure_path_index()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO path_index (abs_path, doc_id, workspace, modified_at) "
                "VALUES (?, ?, ?, ?)",
                (abs_path, doc_id, workspace, modified_at or 0.0),
            )

    def resolve_path(self, abs_path: str) -> dict[str, Any] | None:
        """Look up doc_id by absolute filesystem path."""
        conn = self._ensure_path_index()
        row = conn.execute(
            "SELECT doc_id, workspace, modified_at FROM path_index WHERE abs_path = ?",
            (abs_path,),
        ).fetchone()
        if row:
            return {"doc_id": row[0], "workspace": row[1], "modified_at": row[2]}
        return None

    def delete_from_path_index(self, abs_path: str) -> None:
        """Remove a path entry."""
        conn = self._ensure_path_index()
        with conn:
            conn.execute("DELETE FROM path_index WHERE abs_path = ?", (abs_path,))

    def get_orphaned_paths(self) -> list[str]:
        """Return paths in the index whose local files no longer exist."""
        conn = self._ensure_path_index()
        rows = conn.execute("SELECT abs_path FROM path_index").fetchall()
        return [r[0] for r in rows if not Path(r[0]).exists()]

    def clear_path_index(self, workspace: str | None = None) -> None:
        """Clear all or workspace-specific path mappings."""
        conn = self._ensure_path_index()
        with conn:
            if workspace:
                conn.execute("DELETE FROM path_index WHERE workspace = ?", (workspace,))
            else:
                conn.execute("DELETE FROM path_index")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """ChromaDB restricts collection names to [a-z_0-9]."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower()).strip("_")


def _build_where_clause(
    tags: list[str] | None,
    tags_mode: str,
    extra_filters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compose ChromaDB `where` dict from tags + extra filters."""
    if not tags and not extra_filters:
        return None

    clauses = []
    if tags:
        if tags_mode == "all":
            # ChromaDB doesn't directly support array contains-all
            # Workaround: store tags as flat metadata keys or use "$contains"
            clauses.append({"tags": {"$contains": tags}})
        elif tags_mode == "any":
            clauses.append({"tags": {"$contains": tags}})
        elif tags_mode == "exclude":
            # Exclude mode: no native support, handled at application level
            pass

    if extra_filters:
        clauses.append(extra_filters)

    if len(clauses) == 1:
        return clauses[0]
    if clauses:
        return {"$and": clauses}
    return None


def _normalize_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ChromaDB query results into a list of hit dicts."""
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    ids = raw.get("ids", [[]])[0]
    out = []
    for d, m, dist, rid in zip(docs, metas, dists, ids):
        out.append(
            {
                "chunk_text": d,
                "metadata": m or {},
                "distance": dist,
                "chunk_id": rid,
            }
        )
    return out


def _normalize_get_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ChromaDB get() results."""
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])
    ids = raw.get("ids", [])
    out = []
    for d, m, rid in zip(docs, metas, ids):
        out.append(
            {
                "chunk_id": rid,
                "chunk_text": d,
                "metadata": m or {},
            }
        )
    return out
