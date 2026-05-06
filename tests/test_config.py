"""Tests for Pydantic configuration loading."""

from mcp_rag.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.rag.chunk_size == 600
    assert cfg.rag.chunk_overlap == 60
    assert cfg.memory.lazy_load is True
    assert cfg.memory.idle_ttl_embedder == 300
    assert cfg.tagging.auto_tag_enabled is True
    assert cfg.watcher.debounce_ms == 2000


def test_security_defaults():
    cfg = Config()
    assert cfg.security.ragrules_max_bytes == 102400
    assert cfg.security.yaml_safe_load_only is True


def test_load_config_nonexistent():
    cfg = load_config("/nonexistent/path.yaml")
    assert isinstance(cfg, Config)
    assert cfg.rag.chunk_size == 600
