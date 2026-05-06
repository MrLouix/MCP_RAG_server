"""Main ingestion pipeline — orchestrates extractors, tagging, chunking, embeddings, storage."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

from mcp_rag.chunker import Chunker
from mcp_rag.config import Config
from mcp_rag.embeddings import Embedder
from mcp_rag.extractors import clean_text, extract_file, ExtractedDocument
from mcp_rag.model_manager import ModelManager
from mcp_rag.storage import Storage
from mcp_rag.tagging.engine import TaggingEngine
from mcp_rag.utils.hashing import sha256_file, short_doc_id
from mcp_rag.utils.locks import DocLockRegistry

logger = logging.getLogger(__name__)


class IngestPipeline:
    """End-to-end document ingestion into the vector store."""

    def __init__(
        self,
        config: Config,
        model_manager: ModelManager,
        storage: Storage,
    ) -> None:
        self.cfg = config
        self.mm = model_manager
        self.storage = storage
        self.chunker = Chunker(config.rag.chunk_size, config.rag.chunk_overlap)
        self.embedder = Embedder(model_manager)
        self.tagger = TaggingEngine(config) if config.tagging.auto_tag_enabled else None
        self.locks = DocLockRegistry()

    async def ingest_directory(
        self,
        dir_path: Path,
        recursive: bool = True,
        workspace: str = "default",
    ) -> dict[str, Any]:
        """Ingest all supported files in a directory. Returns stats."""
        t0 = time.time()
        ext_patterns = {e.lower() for e in self.cfg.rag.supported_extensions}
        files = sorted(
            p
            for p in (dir_path.rglob("*") if recursive else dir_path.iterdir())
            if p.is_file() and p.suffix.lower() in ext_patterns and not p.name.startswith(".")
        )

        ingested = 0
        skipped = 0
        errors = 0
        tagging_stats = {"cache_hits": 0, "llm_inferences": 0, "llm_failures": 0, "avg_inference_ms": 0.0}

        # Load OCR once upfront if needed
        ocr_reader = None
        if self.cfg.rag.ocr_enabled:
            try:
                ocr_reader = await self.mm.get_ocr()
            except Exception:
                logger.warning("ocr_load_failed")

        # Process files sequentially with per-docid locking
        for path in files:
            result = await self._ingest_single(path, ocr_reader, workspace, tagging_stats)
            if result["status"] == "ingested":
                ingested += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                errors += 1

        duration_ms = round((time.time() - t0) * 1000, 1)
        return {
            "status": "success",
            "workspace": workspace,
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
            "tagging_stats": tagging_stats,
            "duration_ms": duration_ms,
        }

    async def _ingest_single(
        self,
        path: Path,
        ocr_reader: Any | None,
        workspace: str,
        tagging_stats: dict[str, Any],
    ) -> dict[str, Any]:
        content_hash = sha256_file(path)
        doc_id = short_doc_id(content_hash)

        async with self.locks.acquire(doc_id):
            # Check if already indexed with same hash
            existing = self.storage.resolve_path(str(path))
            if existing and existing.get("doc_id") == doc_id:
                return {"status": "skipped", "reason": "already_indexed", "path": str(path)}

            # Extract
            t_extract = time.time()
            extracted = await asyncio.to_thread(
                extract_file,
                path=path,
                ocr_reader=ocr_reader,
                ocr_enabled=self.cfg.rag.ocr_enabled,
                ocr_languages=self.cfg.rag.ocr_languages,
            )
            if not extracted.text.strip():
                logger.info("empty_extraction", extra={"path": str(path)})
                return {"status": "skipped", "reason": "empty_text", "path": str(path)}

            # Clean
            text = clean_text(extracted.text)

            # Tag
            tags_result = {"system": [], "semantic": [], "llm_status": "disabled"}
            if self.tagger:
                preview = text[:6000]  # ~1500 tokens proxy
                try:
                    tags_result = await self.tagger.tag_document(path, preview, content_hash, self.mm)
                    if tags_result.get("llm_status") == "ok":
                        tagging_stats["llm_inferences"] += 1
                    elif tags_result.get("llm_status") == "timeout":
                        tagging_stats["llm_failures"] += 1
                except Exception as exc:
                    logger.warning("tagging_failed", extra={"path": str(path), "error": str(exc)})

            # Chunk
            chunks = self.chunker.split(text, max_chunks=self.cfg.rag.max_chunks_per_doc)

            # Embed
            model_name = await self.embedder.get_model_name()
            embeddings = await self.embedder.embed(chunks, batch_size=32)

            # Delete old version if re-ingesting same path with different hash
            if existing:
                self.storage.delete_doc_chunks(existing["doc_id"], workspace, model_name)
                self.storage.delete_from_path_index(str(path))

            # Build metadata per chunk
            now = datetime.datetime.utcnow().isoformat() + "Z"
            metadatas = []
            for idx, chunk in enumerate(chunks):
                meta = {
                    "doc_id": doc_id,
                    "source_path": str(path.resolve()),
                    "source_name": path.name,
                    "page_or_section": extracted.pages if extracted.pages else 1,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "ingested_at": now,
                    "modified_at": path.stat().st_mtime,
                    "file_type": extracted.file_type,
                    "content_hash": content_hash,
                    "schema_version": "1.0",
                    "embedding_model": model_name,
                    "orphaned": False,
                    "tags": json.dumps(tags_result, ensure_ascii=False),
                }
                metadatas.append(meta)

            # Store
            self.storage.add_chunks(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                workspace=workspace,
                embedding_model=model_name,
            )
            self.storage.upsert_path_index(str(path.resolve()), doc_id, workspace)

            return {"status": "ingested", "path": str(path), "chunks": len(chunks), "doc_id": doc_id}
