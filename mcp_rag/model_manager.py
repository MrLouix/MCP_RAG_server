"""Lazy-load / idle-unload model manager — the memory cornerstone of the RAG server.

Tracks 3 heavyweight components:
- embedder (sentence-transformers, ~450 MB)
- llm_tagger (llama.cpp, ~2.2 GB)  
- ocr_reader (EasyOCR, ~1.3 GB)

Each is loaded on first demand and unloaded after a configurable idle TTL.
A background GC thread checks every N seconds.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
from sentence_transformers import SentenceTransformer

from mcp_rag.config import Config

logger = logging.getLogger(__name__)


@dataclass
class _ModelSlot:
    name: str
    obj: Any | None = None
    loaded: bool = False
    last_used: float | None = None
    load_duration_ms: float = 0.0
    _load_fn: Any = field(default=None, repr=False)
    _unload_fn: Any = field(default=None, repr=False)


class ModelManager:
    """Central registry for heavyweight ML models with TTL-based eviction."""

    def __init__(self, config: Config) -> None:
        self.cfg = config.memory
        self.rag_cfg = config.rag
        self.tag_cfg = config.tagging

        self._slots: dict[str, _ModelSlot] = {
            "embedder": _ModelSlot(name="embedder"),
            "llm": _ModelSlot(name="llm"),
            "ocr": _ModelSlot(name="ocr"),
        }
        self._lock = asyncio.Lock()
        self._process = psutil.Process()
        self._gc_thread: threading.Thread | None = None
        self._stop_gc = threading.Event()

    # ------------------------------------------------------------------
    # Public async API (safe to call from MCP tools)
    # ------------------------------------------------------------------

    async def get_embedder(self) -> SentenceTransformer:
        """Return embedder, loading on demand."""
        return await self._get("embedder", self._load_embedder)

    async def get_llm(self) -> Any:
        """Return LLM tagger, loading on demand."""
        return await self._get("llm", self._load_llm)

    async def get_ocr(self) -> Any:
        """Return OCR reader, loading on demand."""
        return await self._get("ocr", self._load_ocr)

    async def unload(self, names: list[str] | None = None, force: bool = False) -> dict:
        """Unload models. If names is None, unload all.  Returns freed RAM estimate."""
        names = names or list(self._slots.keys())
        async with self._lock:
            ram_before = self._process.memory_info().rss
            for name in names:
                slot = self._slots.get(name)
                if slot and slot.loaded:
                    await self._unload_slot(slot)
            ram_after = self._process.memory_info().rss
            freed_mb = max(0.0, (ram_after - ram_before) / 1024 / 1024)
            logger.info("model_unload", extra={"unloaded": names, "ram_freed_mb": freed_mb})
        return {"unloaded": names, "ram_freed_mb": freed_mb}

    async def preload(self, names: list[str]) -> dict:
        """Explicitly preload given models (reduces latency before a known batch)."""
        results = {}
        for name in names:
            loader = getattr(self, f"_load_{name}", None)
            if loader:
                await self._get(name, loader)
                results[name] = "loaded"
            else:
                results[name] = "unknown"
        return {"loaded": results}

    async def get_status(self) -> dict:
        """Snapshot of current RAM usage per model."""
        async with self._lock:
            status = {}
            for slot in self._slots.values():
                status[slot.name] = {
                    "loaded": slot.loaded,
                    "last_used_s_ago": round(time.time() - slot.last_used, 1) if slot.last_used else None,
                    "last_loaded_ms": slot.load_duration_ms,
                }
        return status

    # ------------------------------------------------------------------
    # Background GC thread
    # ------------------------------------------------------------------

    def start_gc_thread(self) -> None:
        """Start the daemon GC thread that evicts idle models."""
        if self._gc_thread and self._gc_thread.is_alive():
            return
        self._stop_gc.clear()
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True, name="model_gc")
        self._gc_thread.start()
        logger.info("gc_thread_started", extra={"tick_s": self.cfg.gc_tick_seconds})

    def stop_gc_thread(self) -> None:
        """Signal the GC thread to stop."""
        self._stop_gc.set()
        if self._gc_thread:
            self._gc_thread.join(timeout=2.0)

    def _gc_loop(self) -> None:
        while not self._stop_gc.is_set():
            time.sleep(self.cfg.gc_tick_seconds)
            self._run_gc_round()

    def _run_gc_round(self) -> None:
        """Synchronous entry for GC thread; wraps async calls."""
        try:
            asyncio.run(self.unload_if_idle())
        except Exception as exc:
            logger.warning("gc_round_failed", extra={"error": str(exc)})

    async def unload_if_idle(self) -> dict:
        """Check TTLs and unload stale models. Called by GC thread AND externally."""
        now = time.time()
        stale = []
        async with self._lock:
            for name, slot in self._slots.items():
                if not slot.loaded or slot.last_used is None:
                    continue
                ttl = getattr(self.cfg, f"idle_ttl_{name}", 120)
                if now - slot.last_used > ttl:
                    stale.append(name)
        if stale:
            logger.info("unloading_idle_models", extra={"models": stale, "idle_s": [round(now - self._slots[n].last_used, 1) for n in stale]})
            return await self.unload(stale)
        return {"unloaded": [], "ram_freed_mb": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, name: str, loader: Any) -> Any:
        slot = self._slots[name]
        if slot.loaded and slot.obj is not None:
            slot.last_used = time.time()
            return slot.obj

        async with self._lock:
            # Double-check inside lock
            if slot.loaded and slot.obj is not None:
                slot.last_used = time.time()
                return slot.obj

            t0 = time.time()
            # Run blocking loader in thread pool
            obj = await asyncio.to_thread(loader)
            slot.obj = obj
            slot.loaded = True
            slot.last_used = time.time()
            slot.load_duration_ms = round((time.time() - t0) * 1000, 1)
            logger.info(
                "model_loaded",
                extra={"model": name, "duration_ms": slot.load_duration_ms},
            )
        return obj

    async def _unload_slot(self, slot: _ModelSlot) -> None:
        """Release references and force aggressive cleanup."""
        logger.info("unloading_model", extra={"model": slot.name})
        slot.obj = None
        slot.loaded = False
        slot.last_used = None

        # Try to release PyTorch/CUDA caches if applicable
        try:
            import gc as _gc
            _gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Blocking loaders (run in thread pool via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _load_embedder(self) -> SentenceTransformer:
        model_name = self.rag_cfg.embedding.model
        try:
            return SentenceTransformer(model_name)
        except Exception:
            logger.warning("embedder_fallback", extra={"from": model_name, "to": self.rag_cfg.embedding.fallback})
            return SentenceTransformer(self.rag_cfg.embedding.fallback)

    def _load_llm(self) -> Any:
        from llama_cpp import Llama

        model_path = self.tag_cfg.model_path
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f"LLM model not found at {model_path}")
        return Llama(
            model_path=model_path,
            n_ctx=self.tag_cfg.n_ctx,
            n_threads=self.tag_cfg.n_threads,
            verbose=False,
        )

    def _load_ocr(self) -> Any:
        import easyocr

        langs = self.rag_cfg.ocr_languages or ["fra", "eng"]
        return easyocr.Reader(langs, gpu=False)
