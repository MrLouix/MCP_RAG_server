"""Pydantic configuration model for the MCP RAG server."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")
    backend: str = Field(default="sentence-transformers")
    fallback: str = Field(default="all-MiniLM-L6-v2")


class MemoryConfig(BaseModel):
    lazy_load: bool = Field(default=True)
    idle_ttl_embedder: int = Field(default=300)
    idle_ttl_llm: int = Field(default=120)
    idle_ttl_ocr: int = Field(default=180)
    gc_tick_seconds: int = Field(default=30)
    aggressive_gc: bool = Field(default=True)


class TaggingConfig(BaseModel):
    auto_tag_enabled: bool = Field(default=True)
    model_path: str = Field(default="")
    n_ctx: int = Field(default=4096)
    n_threads: int = Field(default=4)
    timeout_ms: int = Field(default=10000)
    use_cache: bool = Field(default=True)
    cache_path: str = Field(default=".rag_tag_cache.db")
    taxonomy: dict[str, Any] = Field(default_factory=lambda: {
        "domaine": ["financier", "juridique", "technique", "commercial", "rh", "administratif"],
        "priorite": ["urgent", "normal", "faible"],
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
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
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
        yaml_file="config.yaml",
        env_nested_delimiter="__",
    )

    rag: RagConfig = Field(default_factory=RagConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
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
