"""Pydantic configuration model for the MCP RAG server (v4 — Ollama backend)."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaConfig(BaseModel):
    base_url: str = Field(default="http://172.28.128.1:11434")
    embed_model: str = Field(default="nomic-embed-text")
    tag_model: str = Field(default="qwen2.5:3b")
    vision_model: str = Field(default="minicpm-v")
    timeout_s: float = Field(default=30.0)
    embed_timeout_s: float = Field(default=120.0)
    max_retries: int = Field(default=3)
    auto_pull: bool = Field(default=False)


class TaggingConfig(BaseModel):
    auto_tag_enabled: bool = Field(default=True)
    timeout_ms: int = Field(default=30000)
    use_cache: bool = Field(default=True)
    cache_path: str = Field(default=".rag_tag_cache.db")
    taxonomy: dict[str, Any] = Field(default_factory=lambda: {
        "domaine": ["financier", "juridique", "technique", "commercial", "rh", "administratif"],
        "priorite": ["urgent", "normal", "faible"],
        "langue": "ISO 639-1",
        "entites": ["array", "string"],
        "confidentialite": ["public", "interne", "confidentiel"],
    })
    temperature: float = Field(default=0.1)
    seed: int = Field(default=42)


class WatcherConfig(BaseModel):
    enabled: bool = Field(default=False)
    debounce_ms: int = Field(default=2000)
    sync_deletions: bool = Field(default=True)
    max_workers: int = Field(default=4)
    default_watch_paths: list[str] = Field(default_factory=list)
    default_recursive: bool = Field(default=True)


class SecurityConfig(BaseModel):
    ragrules_max_bytes: int = Field(default=102400)
    regex_timeout_ms: int = Field(default=200)
    yaml_safe_load_only: bool = Field(default=True)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: str = Field(default="./logs/mcp_rag.log")
    rotation: str = Field(default="50MB")
    retention_days: int = Field(default=14)


class RagConfig(BaseModel):
    index_path: str = Field(default="./rag_index")
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=60)
    max_chunks_per_doc: int = Field(default=1500)
    ocr_enabled: bool = Field(default=True)
    ocr_languages: list[str] = Field(default_factory=lambda: ["fra", "eng"])
    supported_extensions: list[str] = Field(default_factory=lambda: [
        ".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".markdown",
        ".docx", ".csv", ".xlsx",
    ])


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_RAG_",
        env_nested_delimiter="__",
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    tagging: TaggingConfig = Field(default_factory=TaggingConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("rag", mode="before")
    @classmethod
    def _resolve_index_path(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v.setdefault("index_path", str(Path(__file__).parent.parent / "rag_index"))
        return v


def load_config(path: str | None = None) -> Config:
    """Load configuration from YAML file or defaults."""
    if path and Path(path).exists():
        from yaml import safe_load
        data = safe_load(Path(path).read_text())
        return Config.model_validate(data or {})
    return Config()
