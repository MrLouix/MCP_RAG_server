"""FastMCP server exposing the local RAG index to Hermes Agent.

15 tools: ingest, search, CRUD, watch, tagging, diagnose, model management.
All ML inference delegated to an external Ollama instance.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_rag.config import load_config, Config
from mcp_rag.ingest import IngestPipeline
from mcp_rag.logging_config import setup_logging
from mcp_rag.ollama_client import OllamaClient
from mcp_rag.storage import Storage
from mcp_rag.watcher.fs_watcher import WatchManager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global state (initialized in main())
# ------------------------------------------------------------------
_g_config: Config | None = None
_g_ollama: OllamaClient | None = None
_g_storage: Storage | None = None
_g_pipeline: IngestPipeline | None = None
_g_watcher: WatchManager | None = None

_mcp = FastMCP(
    "RAG Documents",
    instructions="Local RAG server with auto-tagging, watch folder, and Ollama backend.",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _init_globals(config_path: str | None = None) -> None:
    """One-shot initialization of server globals."""
    global _g_config, _g_ollama, _g_storage, _g_pipeline, _g_watcher
    _g_config = load_config(config_path)
    setup_logging(
        level=_g_config.logging.level,
        fmt=_g_config.logging.format,
        file_path=_g_config.logging.file,
    )
    _g_ollama = OllamaClient(
        base_url=_g_config.ollama.base_url,
        timeout_s=_g_config.ollama.timeout_s,
        embed_timeout_s=_g_config.ollama.embed_timeout_s,
        max_retries=_g_config.ollama.max_retries,
    )
    _g_storage = Storage(_g_config.rag.index_path)
    _g_pipeline = IngestPipeline(_g_config, _g_ollama, _g_storage)
    _g_watcher = WatchManager(_g_config)
    _g_watcher.inject_dependencies(_g_pipeline, _g_storage, _g_ollama)
    logger.info("server_initialized", extra={"config": config_path or "default"})


# ------------------------------------------------------------------
# 1. ingest_directory
# ------------------------------------------------------------------

@_mcp.tool()
async def ingest_directory(
    dir_path: str,
    recursive: bool = True,
    workspace: str = "default",
) -> dict[str, Any]:
    """Index all supported files in a directory with auto-tagging."""
    if _g_pipeline is None:
        return {"error": "Server not initialized"}
    path = Path(dir_path).expanduser().resolve()
    if not path.exists():
        return {"error": f"Directory not found: {dir_path}"}
    result = await _g_pipeline.ingest_directory(path, recursive, workspace)
    return result


# ------------------------------------------------------------------
# 2. search_docs
# ------------------------------------------------------------------

@_mcp.tool()
async def search_docs(
    query: str,
    tags: list[str] | None = None,
    tags_mode: str = "any",
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    workspace: str = "default",
) -> dict[str, Any]:
    """Semantic search with optional tag filtering."""
    if _g_ollama is None or _g_storage is None or _g_config is None:
        return {"error": "Server not initialized"}

    from mcp_rag.embeddings import Embedder
    embedder = Embedder(_g_ollama, _g_config.ollama.embed_model)
    query_vec = await embedder.embed([query], batch_size=1)
    model_name = await embedder.get_model_name()

    # Broad semantic search (tag filtering at application level)
    raw_results = _g_storage.search(
        query_embedding=query_vec[0],
        top_k=top_k * 3,
        tags=None,
        filters=filters,
        workspace=workspace,
        embedding_model=model_name,
    )

    # Filter by tags at application level
    filtered = []
    if tags:
        for hit in raw_results:
            meta = hit.get("metadata", {})
            tags_data = json.loads(meta.get("tags", "{}"))
            merged = tags_data.get("system", []) + tags_data.get("semantic", [])
            if tags_mode == "all":
                ok = all(t in merged for t in tags)
            elif tags_mode == "any":
                ok = any(t in merged for t in tags)
            elif tags_mode == "exclude":
                ok = not any(t in merged for t in tags)
            else:
                ok = True
            if ok:
                filtered.append(hit)
    else:
        filtered = raw_results

    return {"results": filtered[:top_k], "query": query, "workspace": workspace}


# ------------------------------------------------------------------
# 3. get_document
# ------------------------------------------------------------------

@_mcp.tool()
async def get_document(
    doc_id: str,
    workspace: str = "default",
) -> dict[str, Any]:
    """Retrieve all chunks for a specific document."""
    if _g_storage is None:
        return {"error": "Server not initialized"}
    chunks = _g_storage.get_document_chunks(doc_id, workspace)
    return {"doc_id": doc_id, "chunks": chunks, "workspace": workspace}


# ------------------------------------------------------------------
# 4. list_documents
# ------------------------------------------------------------------

@_mcp.tool()
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    tags: list[str] | None = None,
    tags_mode: str = "all",
    include_orphaned: bool = False,
    workspace: str = "default",
) -> dict[str, Any]:
    """List indexed documents with optional tag filtering."""
    if _g_storage is None:
        return {"error": "Server not initialized"}

    doc_ids = sorted(_g_storage.list_doc_ids(workspace))
    total = len(doc_ids)

    paginated = doc_ids[offset:offset + limit]

    documents = []
    for did in paginated:
        meta = _g_storage.get_document_chunks(did, workspace)
        if meta:
            first = meta[0]
            m = first.get("metadata", {})
            tags_data = json.loads(m.get("tags", "{}"))
            documents.append({
                "doc_id": did,
                "name": m.get("source_name", ""),
                "type": m.get("file_type", ""),
                "chunks_count": m.get("total_chunks", 0),
                "ingested_at": m.get("ingested_at", ""),
                "tags": tags_data,
                "orphaned": m.get("orphaned", False),
            })

    # Filter by tags at application level
    if tags:
        filtered = []
        for doc in documents:
            tags_data = doc.get("tags", {})
            merged = tags_data.get("system", []) + tags_data.get("semantic", [])
            if tags_mode == "all":
                ok = all(t in merged for t in tags)
            elif tags_mode == "any":
                ok = any(t in merged for t in tags)
            elif tags_mode == "exclude":
                ok = not any(t in merged for t in tags)
            else:
                ok = True
            if ok:
                filtered.append(doc)
        documents = filtered

    if not include_orphaned:
        documents = [d for d in documents if not d.get("orphaned")]

    return {
        "documents": documents,
        "total_count": total,
        "returned": len(documents),
        "workspace": workspace,
    }


# ------------------------------------------------------------------
# 5. delete_document
# ------------------------------------------------------------------

@_mcp.tool()
async def delete_document(
    doc_id: str,
    workspace: str = "default",
) -> dict[str, Any]:
    """Remove a document and all its chunks from the index."""
    if _g_storage is None or _g_config is None:
        return {"error": "Server not initialized"}
    model_name = _g_config.ollama.embed_model
    _g_storage.delete_doc_chunks(doc_id, workspace, model_name)
    _g_storage.delete_from_path_index_by_doc_id(doc_id)
    return {"status": "deleted", "doc_id": doc_id, "workspace": workspace}


# ------------------------------------------------------------------
# 6. clear_index
# ------------------------------------------------------------------

@_mcp.tool()
async def clear_index(
    confirm: bool = False,
    workspace: str = "default",
) -> dict[str, Any]:
    """Delete all chunks in a workspace. Requires confirm=True."""
    if not confirm:
        return {"error": "confirm must be True to clear the index"}
    if _g_storage is None:
        return {"error": "Server not initialized"}
    _g_storage.delete_collection(workspace)
    _g_storage.clear_path_index(workspace)
    return {"status": "cleared", "workspace": workspace}


# ------------------------------------------------------------------
# 7. get_stats
# ------------------------------------------------------------------

@_mcp.tool()
async def get_stats(
    workspace: str = "default",
) -> dict[str, Any]:
    """Get current index statistics."""
    if _g_storage is None or _g_config is None:
        return {"error": "Server not initialized"}
    total_docs = len(_g_storage.list_doc_ids(workspace))
    total_chunks = _g_storage.count_chunks(workspace)
    model_status = {}
    if _g_ollama:
        try:
            model_status = await _g_ollama.healthcheck()
        except Exception:
            model_status = {"reachable": False}
    watcher_info = []
    if _g_watcher:
        watcher_info = _g_watcher.list_active()
    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "embedding_model": _g_config.ollama.embed_model,
        "tag_model": _g_config.ollama.tag_model,
        "active_watchers": watcher_info,
        "model_status": model_status,
        "workspace": workspace,
    }


# ------------------------------------------------------------------
# 8. watch_directory
# ------------------------------------------------------------------

@_mcp.tool()
async def watch_directory(
    dir_path: str,
    recursive: bool = True,
    enabled: bool = True,
    sync_deletions: bool = True,
    debounce_ms: int = 2000,
    workspace: str = "default",
) -> dict[str, Any]:
    """Activate or deactivate filesystem watching for a directory."""
    if _g_watcher is None:
        return {"error": "Server not initialized"}
    if enabled:
        if _g_watcher._queue is None:
            loop = asyncio.get_running_loop()
            _g_watcher.start(loop)
            logger.info("watcher_started_lazy")
        wid = _g_watcher.add_watch(str(Path(dir_path).resolve()), recursive)
        return {
            "status": "watching",
            "watcher_id": wid,
            "watched_paths": [p for p in _g_watcher._observers],
            "recursive": recursive,
            "sync_deletions": sync_deletions,
        }
    else:
        ok = _g_watcher.remove_watch(str(Path(dir_path).resolve()))
        return {
            "status": "stopped" if ok else "not_found",
            "watched_paths": [p for p in _g_watcher._observers],
        }


# ------------------------------------------------------------------
# 9. get_tags
# ------------------------------------------------------------------

@_mcp.tool()
async def get_tags(
    query: str | None = None,
    workspace: str = "default",
) -> dict[str, Any]:
    """Return all tags with counts across the workspace."""
    if _g_storage is None:
        return {"error": "Server not initialized"}
    docs = await list_documents(limit=10000, workspace=workspace)
    tag_counts: dict[str, dict[str, Any]] = {}
    for doc in docs.get("documents", []):
        tags_data = doc.get("tags", {})
        for origin in ("system", "semantic"):
            for tag in tags_data.get(origin, []):
                if tag not in tag_counts:
                    tag_counts[tag] = {"tag": tag, "count": 0, "origin": origin}
                tag_counts[tag]["count"] += 1
    tags_list = list(tag_counts.values())
    if query:
        query_lower = query.lower()
        tags_list = [t for t in tags_list if query_lower in t["tag"].lower()]
    return {"tags": tags_list, "total_tags": len(tag_counts)}


# ------------------------------------------------------------------
# 10. tag_document
# ------------------------------------------------------------------

@_mcp.tool()
async def tag_document(
    doc_id: str,
    force_retag: bool = False,
    workspace: str = "default",
) -> dict[str, Any]:
    """Retag a single document without full re-ingestion."""
    if _g_storage is None or _g_ollama is None or _g_config is None:
        return {"error": "Server not initialized"}
    chunks = _g_storage.get_document_chunks(doc_id, workspace)
    if not chunks:
        return {"error": "Document not found", "doc_id": doc_id}

    from mcp_rag.tagging.heuristics import HeuristicTagger

    meta = chunks[0].get("metadata", {})
    source_path = meta.get("source_path", "")
    old_tags = json.loads(meta.get("tags", "{}"))

    if not force_retag:
        return {"status": "success", "doc_id": doc_id, "tags": old_tags, "duration_ms": 0, "cache_hit": True}

    # Recalculate H1 tags
    h1 = HeuristicTagger().tag_document(Path(source_path))
    new_tags = {
        "system": h1,
        "semantic": old_tags.get("semantic", []),
        "model": old_tags.get("model", ""),
        "inferred_at": old_tags.get("inferred_at", ""),
        "llm_status": "disabled",
    }

    # Update all chunks in ChromaDB
    coll = _g_storage.get_collection(workspace)
    ids_to_update = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    new_metas = [dict(c.get("metadata", {}), tags=json.dumps(new_tags)) for c in chunks if c.get("chunk_id")]
    if ids_to_update and new_metas:
        coll.update(ids=ids_to_update, metadatas=list(new_metas))

    return {"status": "success", "doc_id": doc_id, "tags": new_tags, "duration_ms": 0, "cache_hit": False, "new_system_tags": h1}


# ------------------------------------------------------------------
# 11. reindex_all
# ------------------------------------------------------------------

@_mcp.tool()
async def reindex_all(
    new_embedding_model: str,
    confirm: bool = False,
    workspace: str = "default",
) -> dict[str, Any]:
    """Reindex all documents with a new embedding model via Ollama."""
    if not confirm:
        return {"error": "confirm must be True to reindex"}
    # TODO: implement full reindex with path_index traversal
    return {
        "status": "not_yet_implemented",
        "previous_model": "",
        "new_model": new_embedding_model,
        "reindexed": 0,
    }


# ------------------------------------------------------------------
# 12. diagnose
# ------------------------------------------------------------------

@_mcp.tool()
async def diagnose(
    workspace: str = "default",
) -> dict[str, Any]:
    """Run a healthcheck on the RAG system (Ollama, ChromaDB, disk, watchers)."""
    import shutil
    warnings: list[str] = []

    # Ollama
    ollama_status: dict[str, Any] = {"reachable": False, "base_url": ""}
    if _g_ollama and _g_config:
        required = [
            _g_config.ollama.embed_model,
            _g_config.ollama.tag_model,
            _g_config.ollama.vision_model,
        ]
        ollama_status = await _g_ollama.healthcheck(required_models=required)
        if not ollama_status.get("reachable"):
            warnings.append("Ollama instance unreachable")
        missing = ollama_status.get("missing_models", [])
        if missing:
            warnings.append(f"Missing Ollama models: {', '.join(missing)}")

    # ChromaDB
    collections: list[str] = []
    chroma_ok = False
    if _g_storage is not None:
        try:
            collections = _g_storage.list_collections()
            chroma_ok = True
        except Exception:
            chroma_ok = False

    # Disk
    index_size_mb = 0.0
    free_gb = 0.0
    try:
        index_path = Path(_g_config.rag.index_path) if _g_config else Path("./rag_index")
        if index_path.exists():
            total_size = sum(f.stat().st_size for f in index_path.rglob("*") if f.is_file())
            index_size_mb = round(total_size / (1024 * 1024), 2)
        free = shutil.disk_usage(str(index_path.resolve().parent if index_path.exists() else "."))
        free_gb = round(free.free / (1024**3), 2)
    except Exception:
        pass

    # Orphans
    orphans: list[str] = []
    if _g_storage:
        orphans = _g_storage.get_orphaned_paths()
        if orphans:
            warnings.append(f"{len(orphans)} orphaned paths detected")

    # Watchers
    watchers: list[dict[str, Any]] = []
    if _g_watcher:
        watchers = _g_watcher.list_active()

    return {
        "status": "ok" if not warnings else "warning",
        "ollama": ollama_status,
        "chroma": {"reachable": chroma_ok, "collections": collections},
        "disk": {"index_size_mb": index_size_mb, "free_gb": free_gb, "warning": None},
        "watchers": watchers,
        "orphans": {"count": len(orphans), "examples": orphans[:5]},
        "warnings": warnings,
    }


# ------------------------------------------------------------------
# 13/14/15. Model management tools (delegated to Ollama)
# ------------------------------------------------------------------

@_mcp.tool()
async def get_model_status() -> dict[str, Any]:
    """Show which ML models are currently loaded on the Ollama instance."""
    if _g_ollama is None or _g_config is None:
        return {"error": "Server not initialized"}
    try:
        running = await _g_ollama.list_running()
    except Exception as exc:
        return {"error": f"Ollama unreachable: {exc}", "ollama_url": _g_config.ollama.base_url}
    return {
        "ollama_url": _g_config.ollama.base_url,
        "ollama_reachable": True,
        "models_running": [
            {"name": m.get("name", ""), "size_mb": round(m.get("size", 0) / (1024 * 1024), 1)}
            for m in running
        ],
        "embed_model": _g_config.ollama.embed_model,
        "tag_model": _g_config.ollama.tag_model,
        "vision_model": _g_config.ollama.vision_model,
    }


@_mcp.tool()
async def unload_models(
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Ask Ollama to unload models from memory (keep_alive=0)."""
    if _g_ollama is None or _g_config is None:
        return {"error": "Server not initialized"}
    targets = models or [
        _g_config.ollama.embed_model,
        _g_config.ollama.tag_model,
        _g_config.ollama.vision_model,
    ]
    unloaded = []
    for m in targets:
        try:
            await _g_ollama.unload(m)
            unloaded.append(m)
        except Exception as exc:
            logger.warning("unload_failed", extra={"model": m, "error": str(exc)})
    return {"unloaded": unloaded, "status": "ok"}


@_mcp.tool()
async def preload_models(
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Preload models on the Ollama instance to reduce first-hit latency."""
    if _g_ollama is None or _g_config is None:
        return {"error": "Server not initialized"}
    targets = models or [_g_config.ollama.embed_model]
    loaded = []
    for m in targets:
        try:
            await _g_ollama.preload(m)
            loaded.append(m)
        except Exception as exc:
            logger.warning("preload_failed", extra={"model": m, "error": str(exc)})
    return {"loaded": loaded, "status": "ok"}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MCP RAG Server")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default=os.environ.get("RAG_TRANSPORT", "stdio"),
        help="MCP transport protocol",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "3000")), help="SSE port"
    )
    parser.add_argument(
        "--host", type=str, default=os.environ.get("HOST", "127.0.0.1"), help="SSE host"
    )
    args = parser.parse_args()

    _init_globals(args.config)

    # Start watcher event loop consumer if enabled
    if _g_watcher and _g_config and _g_config.watcher.enabled:
        loop = asyncio.get_event_loop()
        _g_watcher.start(loop)
        for p in _g_config.watcher.default_watch_paths:
            _g_watcher.add_watch(p, recursive=_g_config.watcher.default_recursive)

    logger.info("server_starting", extra={"transport": args.transport})
    if args.transport == "sse":
        import uvicorn
        app = _mcp.streamable_http_app()
        logger.info("streamable_http_server_starting", extra={"port": args.port, "host": args.host})
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    else:
        _mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
