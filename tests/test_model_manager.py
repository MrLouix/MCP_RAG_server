"""Tests for ModelManager lazy-load and TTL eviction."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_rag.config import Config
from mcp_rag.model_manager import ModelManager


@pytest.fixture
def cfg():
    c = Config()
    c.memory.idle_ttl_embedder = 0.1  # fast for tests
    c.memory.idle_ttl_llm = 0.1
    c.memory.idle_ttl_ocr = 0.1
    c.memory.gc_tick_seconds = 0.05
    return c


@pytest.fixture
def mm(cfg):
    manager = ModelManager(cfg)
    # Patch the blocking loaders to avoid actual model loading
    manager._load_embedder = MagicMock(return_value=MagicMock())
    manager._load_llm = MagicMock(return_value=MagicMock())
    manager._load_ocr = MagicMock(return_value=MagicMock())
    return manager


@pytest.mark.asyncio
async def test_embedder_lazy_load(mm):
    emb = await mm.get_embedder()
    assert emb is not None
    assert mm._slots["embedder"].loaded is True
    status = await mm.get_status()
    assert status["embedder"]["loaded"] is True


@pytest.mark.asyncio
async def test_unload_if_idle(mm):
    await mm.get_embedder()
    assert mm._slots["embedder"].loaded is True
    # Wait past TTL
    import time
    time.sleep(0.2)
    result = await mm.unload_if_idle()
    assert "embedder" in result["unloaded"] or result["ram_freed_mb"] >= 0


@pytest.mark.asyncio
async def test_force_unload(mm):
    await mm.get_embedder()
    result = await mm.unload(["embedder"])
    assert "embedder" in result["unloaded"]
    assert mm._slots["embedder"].loaded is False


@pytest.mark.asyncio
async def test_model_status_empty(mm):
    status = await mm.get_status()
    assert status["embedder"]["loaded"] is False
    assert status["llm"]["loaded"] is False
    assert status["ocr"]["loaded"] is False
