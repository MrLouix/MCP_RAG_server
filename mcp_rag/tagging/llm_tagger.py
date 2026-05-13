"""Couche H2 — tags semantiques inferes par un LLM via Ollama /api/chat (JSON mode).

- JSON Mode natif Ollama (format: "json")
- Cache SQLite par content_hash
- Timeout strict pour ne pas bloquer MCP stdio
- Serialisation des appels via asyncio.Semaphore(1)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY = {
    "domaine": ["financier", "juridique", "technique", "commercial", "rh", "administratif"],
    "priorite": ["urgent", "normal", "faible"],
    "langue": "ISO 639-1",
    "entites": ["array", "string"],
    "confidentialite": ["public", "interne", "confidentiel"],
}


class TagCache:
    """SQLite-backed tag cache keyed by content hash."""

    def __init__(self, db_path: str = ".rag_tag_cache.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tag_cache (
                content_hash TEXT PRIMARY KEY,
                tags_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                inferred_at REAL NOT NULL
            )
            """
        )

    def get(self, content_hash: str, current_model_version: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT tags_json, model_version FROM tag_cache WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if not row:
            return None
        tags_json, stored_version = row
        if stored_version != current_model_version:
            return None
        return json.loads(tags_json)

    def set(
        self,
        content_hash: str,
        tags: dict[str, Any],
        model_version: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO tag_cache (content_hash, tags_json, model_version, inferred_at) "
                "VALUES (?, ?, ?, ?)",
                (content_hash, json.dumps(tags, ensure_ascii=False), model_version, time.time()),
            )


class LLMTagger:
    """Semantic tagger using Ollama /api/chat with JSON mode."""

    def __init__(self, config: Any) -> None:
        self.cfg = config
        self.taxonomy = config.taxonomy or DEFAULT_TAXONOMY
        self.cache = TagCache(config.cache_path)
        # Serialize LLM calls to avoid contention on the Ollama instance
        self._semaphore = asyncio.Semaphore(1)

    async def tag_document(
        self,
        text_preview: str,
        file_name: str,
        h1_tags: list[str],
        content_hash: str,
        ollama_client: Any,
        tag_model: str,
    ) -> dict[str, Any]:
        """Infer semantic tags via Ollama. Returns empty dict on failure."""
        model_version = self._model_version(tag_model)
        cached = self.cache.get(content_hash, model_version)
        if cached:
            return {"cached": True, "llm_status": "ok", **cached}

        prompt = self._build_prompt(file_name, h1_tags, text_preview)
        t0 = time.time()

        try:
            async with self._semaphore:
                output = await asyncio.wait_for(
                    ollama_client.chat(
                        model=tag_model,
                        messages=[{"role": "user", "content": prompt}],
                        format="json",
                        options={
                            "temperature": self.cfg.temperature,
                            "seed": self.cfg.seed,
                            "num_predict": 256,
                        },
                    ),
                    timeout=self.cfg.timeout_ms / 1000.0,
                )
            elapsed_ms = round((time.time() - t0) * 1000, 1)
        except asyncio.TimeoutError:
            logger.warning("llm_inference_timeout", extra={"file": file_name, "timeout_ms": self.cfg.timeout_ms})
            return {"semantic": [], "llm_status": "timeout"}
        except Exception as exc:
            logger.warning("llm_inference_failed", extra={"file": file_name, "error": str(exc)})
            return {"semantic": [], "llm_status": "error", "error": str(exc)}

        tags = self._parse_output(output)
        tags["llm_status"] = "ok"
        tags["inference_ms"] = elapsed_ms
        self.cache.set(content_hash, tags, model_version)
        return tags

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _model_version(self, tag_model: str) -> str:
        """Hash of model name + taxonomy keys for cache invalidation."""
        from mcp_rag.utils.hashing import sha256_bytes

        data = f"{tag_model}:{sorted((self.taxonomy or {}).keys())}".encode()
        return sha256_bytes(data)[:16]

    def _build_prompt(self, file_name: str, h1_tags: list[str], text_preview: str) -> str:
        taxonomy_text = json.dumps(self.taxonomy, ensure_ascii=False, indent=2)
        h1_tags_text = ", ".join(h1_tags) if h1_tags else "aucun"
        return (
            f"Tu es un classifieur de documents. Analyse le document suivant et reponds "
            f"UNIQUEMENT par un objet JSON valide respectant exactement le schema:\n"
            f"{taxonomy_text}\n\n"
            f"Regles:\n"
            f"- Ne produis aucun texte hors du JSON.\n"
            f"- Si l'information est absente, utilise null.\n\n"
            f"Fichier : {file_name}\n"
            f"Tags connus : {h1_tags_text}\n\n"
            f"Document :\n"
            f"--- Debut ---\n"
            f"{text_preview[:2000]}\n"
            f"--- Fin ---"
        )

    @staticmethod
    def _parse_output(text: str) -> dict[str, Any]:
        """Extract JSON from LLM output."""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            raw_json = text[start:end]
            parsed = json.loads(raw_json)
            return {"semantic": _flatten_tags(parsed)}
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("llm_json_parse_failed", extra={"output_preview": text[:200], "error": str(exc)})
            return {"semantic": [], "json_error": str(exc)}


def _flatten_tags(parsed: dict[str, Any]) -> list[str]:
    """Flatten {domaine: "financier", priorite: null} -> ["domaine:financier", ...]."""
    tags = []
    for key, value in parsed.items():
        if value is None:
            continue
        if isinstance(value, list):
            for v in value:
                if v:
                    tags.append(f"{key}:{str(v).lower()}")
        else:
            tags.append(f"{key}:{str(value).lower()}")
    return tags
