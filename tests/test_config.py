"""Tests for Pydantic configuration loading."""

from mcp_rag.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.rag.chunk_size == 600
    assert cfg.rag.chunk_overlap == 60
    assert cfg.ollama.base_url == "http://172.28.128.1:11434"
    assert cfg.ollama.embed_model == "nomic-embed-text"
    assert cfg.ollama.tag_model == "qwen2.5:3b"
    assert cfg.ollama.vision_model == "minicpm-v"
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


def test_ollama_config_defaults():
    cfg = Config()
    assert cfg.ollama.timeout_s == 30.0
    assert cfg.ollama.embed_timeout_s == 120.0
    assert cfg.ollama.max_retries == 3
    assert cfg.ollama.auto_pull is False


def test_tagging_taxonomy_includes_langue_entites():
    cfg = Config()
    assert "langue" in cfg.tagging.taxonomy
    assert "entites" in cfg.tagging.taxonomy
