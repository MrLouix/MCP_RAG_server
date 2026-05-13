"""Filesystem watcher — watchdog with debounce and async event queue.

Watches directories for create/modify/move/delete events,
debounces rapid changes, and dispatches to the ingest pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _RagFileEventHandler(FileSystemEventHandler):
    """Internal watchdog handler that buffers events with debounce."""

    def __init__(
        self,
        queue: asyncio.Queue,
        event_loop: asyncio.AbstractEventLoop,
        debounce_ms: int,
        supported_extensions: set[str],
    ) -> None:
        self.queue = queue
        self.loop = event_loop
        self.debounce_s = debounce_ms / 1000.0
        self.supported_extensions = supported_extensions
        self._last_event: dict[str, float] = {}  # path -> timestamp
        self._lock = threading.Lock()

    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.supported_extensions

    def _should_emit(self, path: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last_event.get(path, 0)
            if now - last > self.debounce_s:
                self._last_event[path] = now
                return True
            self._last_event[path] = now
            return False

    def _put_event(self, event: dict[str, Any]) -> None:
        """Schedule event into the async queue from a watchdog thread."""
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(event), loop=self.loop)
        except Exception as exc:
            logger.warning("queue_put_failed", extra={"error": str(exc), "event": event})

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.src_path):
            if self._should_emit(event.src_path):
                self._put_event({
                    "type": "created",
                    "path": event.src_path,
                    "time": time.time(),
                })

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.src_path):
            if self._should_emit(event.src_path):
                self._put_event({
                    "type": "modified",
                    "path": event.src_path,
                    "time": time.time(),
                })

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.src_path):
            self._put_event({
                "type": "deleted",
                "path": event.src_path,
                "time": time.time(),
            })

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.dest_path):
            self._put_event({
                "type": "moved",
                "src": event.src_path,
                "dest": event.dest_path,
                "time": time.time(),
            })


class WatchManager:
    """Manages watchdog observers and async event dispatch."""

    def __init__(self, config: Any) -> None:
        self.cfg = config.watcher
        self.rag_cfg = config.rag
        self._observers: dict[str, Observer] = {}
        self._queue: asyncio.Queue | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        # Dependencies injected by server
        self._pipeline: Any | None = None
        self._storage: Any | None = None
        self._ollama: Any | None = None

    def inject_dependencies(
        self,
        pipeline: Any,
        storage: Any,
        ollama_client: Any,
    ) -> None:
        """Inject server globals so the consumer can access them."""
        self._pipeline = pipeline
        self._storage = storage
        self._ollama = ollama_client

    def start(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = event_loop
        self._queue = asyncio.Queue()
        # Schedule the consumer coroutine on the provided event loop
        self._event_loop.create_task(self._consumer())

    async def _consumer(self) -> None:
        """Background coroutine that consumes filesystem events and ingests files."""
        logger.info("watcher_consumer_started")
        while True:
            event = await self._queue.get()
            try:
                event_type = event.get("type")
                path = event.get("path")
                logger.info("watcher_event_queued", extra={
                    "type": event_type,
                    "path": path,
                    "time": event.get("time"),
                })
                await self._handle_event(event)
            except Exception as exc:
                logger.warning("watcher_event_handler_failed", extra={
                    "error": str(exc),
                    "event": event,
                })
            finally:
                self._queue.task_done()

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Process a single filesystem event by delegating to the ingestion pipeline."""
        event_type = event.get("type")
        path_str = event.get("path", "")

        if event_type in ("created", "modified"):
            if self._pipeline is None:
                logger.warning("watcher_pipeline_not_available")
                return

            p = Path(path_str)
            if not p.exists():
                logger.info("watcher_file_gone", extra={"path": path_str})
                return
            if p.suffix.lower() not in {e.lower() for e in self.rag_cfg.supported_extensions}:
                return

            try:
                logger.info("watcher_ingesting_file", extra={"path": path_str})

                tagging_stats: dict[str, Any] = {
                    "cache_hits": 0, "llm_inferences": 0,
                    "llm_failures": 0, "avg_inference_ms": 0.0,
                }
                result = await self._pipeline._ingest_single(p, "default", tagging_stats)
                logger.info("watcher_file_ingested", extra={
                    "path": path_str,
                    "status": result.get("status"),
                    "doc_id": result.get("doc_id"),
                    "chunks": result.get("chunks"),
                })
            except Exception as exc:
                logger.warning("watcher_ingest_failed", extra={
                    "path": path_str,
                    "error": str(exc),
                })

        elif event_type == "deleted":
            if self._storage is not None and self.cfg.sync_deletions:
                existing = self._storage.resolve_path(path_str)
                if existing:
                    doc_id = existing["doc_id"]
                    self._storage.delete_doc_chunks(doc_id, "default")
                    self._storage.delete_from_path_index(path_str)
                    logger.info("watcher_file_deleted_from_index", extra={
                        "path": path_str,
                        "doc_id": doc_id,
                    })

        elif event_type == "moved":
            src = event.get("src", "")
            dest = event.get("dest", "")
            # Remove old entry
            if self._storage is not None:
                existing = self._storage.resolve_path(src)
                if existing:
                    self._storage.delete_doc_chunks(existing["doc_id"], "default")
                    self._storage.delete_from_path_index(src)
            # Ingest the new location
            if Path(dest).suffix.lower() in {e.lower() for e in self.rag_cfg.supported_extensions}:
                await self._handle_event({"type": "created", "path": dest, "time": event.get("time")})

    @property
    def queue(self) -> asyncio.Queue | None:
        return self._queue

    def add_watch(self, path: str, recursive: bool = True) -> str:
        """Add a directory to watch. Returns watcher_id."""
        resolved = Path(path).resolve()
        if not resolved.exists():
            logger.warning("watch_path_not_found", extra={"path": path})
            return f"wd_{path}_not_found"
        if not resolved.is_dir():
            logger.warning("watch_path_not_dir", extra={"path": path})
            return f"wd_{path}_not_dir"

        watched_path = str(resolved)
        if watched_path in self._observers:
            return f"wd_{watched_path}"

        if self._queue is None:
            raise RuntimeError("WatchManager not started")
        if self._event_loop is None:
            raise RuntimeError("WatchManager event loop not set")

        handler = _RagFileEventHandler(
            queue=self._queue,
            event_loop=self._event_loop,
            debounce_ms=self.cfg.debounce_ms,
            supported_extensions={e.lower() for e in self.rag_cfg.supported_extensions},
        )
        observer = Observer()
        observer.schedule(handler, path=watched_path, recursive=recursive)
        observer.start()
        self._observers[watched_path] = observer
        logger.info("watch_started", extra={"path": watched_path, "recursive": recursive})
        return f"wd_{watched_path}"

    def remove_watch(self, path: str) -> bool:
        """Stop watching a directory."""
        observer = self._observers.pop(path, None)
        if observer:
            observer.stop()
            observer.join(timeout=3.0)
            logger.info("watch_stopped", extra={"path": path})
            return True
        return False

    def stop_all(self) -> None:
        """Stop all observers."""
        for path, observer in list(self._observers.items()):
            observer.stop()
            observer.join(timeout=3.0)
            del self._observers[path]
        logger.info("all_watches_stopped")

    def list_active(self) -> list[dict[str, Any]]:
        """Return list of active watchers."""
        return [{"watcher_id": f"wd_{p}", "path": p} for p in self._observers]
