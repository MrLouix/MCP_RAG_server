"""Couche H2 — tags sémantiques inférés par un LLM local (llama.cpp).

- JSON Mode via GBNF grammar
- Cache SQLite par content_hash
- Timeout strict pour ne pas bloquer MCP stdio
"""

from __future__ import annotations

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
            # Model changed → invalidate
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
    """Local LLM-based semantic tagger using llama.cpp."""

    def __init__(self, config: Any) -> None:
        self.cfg = config
        self.taxonomy = config.taxonomy or DEFAULT_TAXONOMY
        self.cache = TagCache(config.cache_path)

    async def tag_document(
        self,
        text_preview: str,
        file_name: str,
        h1_tags: list[str],
        content_hash: str,
        model_manager: Any,
    ) -> dict[str, Any]:
        """Infer semantic tags. Returns empty dict on failure."""
        model_version = self._model_version()
        cached = self.cache.get(content_hash, model_version)
        if cached:
            return {"cached": True, **cached}

        try:
            llm = await model_manager.get_llm()
        except Exception as exc:
            logger.warning("llm_load_failed", extra={"error": str(exc)})
            return {"llm_status": "error", "error": str(exc)}

        prompt = self._build_prompt(file_name, h1_tags, text_preview)
        t0 = time.time()

        try:
            output = await self._infer(llm, prompt)
            elapsed_ms = round((time.time() - t0) * 1000, 1)
        except Exception as exc:
            logger.warning("llm_inference_timeout" if "timeout" in str(exc).lower() else "llm_inference_failed", extra={"error": str(exc)})
            return {"llm_status": "timeout", "error": str(exc)}

        tags = self._parse_output(output)
        tags["llm_status"] = "ok"
        tags["inference_ms"] = elapsed_ms
        self.cache.set(content_hash, tags, model_version)
        return tags

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _model_version(self) -> str:
        """Hash of model path + taxonomy keys for cache invalidation."""
        from mcp_rag.utils.hashing import sha256_bytes

        data = f"{self.cfg.model_path}:{sorted((self.taxonomy or {}).keys())}".encode()
        return sha256_bytes(data)[:16]

    def _build_prompt(self, file_name: str, h1_tags: list[str], text_preview: str) -> str:
        taxonomy_text = json.dumps(self.taxonomy, ensure_ascii=False, indent=2)
        h1_tags_text = ", ".join(h1_tags) if h1_tags else "aucun"
        return (
            f"Tu es un classifieur de documents. Analyse le document suivant et réponds "
            f"UNIQUEMENT par un objet JSON valide respectant exactement le schéma:\n"
            f"{taxonomy_text}\n\n"
            f"Règles:\n"
            f"- Ne produis aucun texte hors du JSON.\n"
            f"- Si l'information est absente, utilise null.\n\n"
            f"Fichier : {file_name}\n"
            f"Tags connus : {h1_tags_text}\n\n"
            f"Document :\n"
            f"--- Début ---\n"
            f"{text_preview[:2000]}\n"
            f"--- Fin ---"
        )

    @staticmethod
    async def _infer(llm: Any, prompt: str) -> str:
        """Run LLM inference via llama.cpp."""
        result = llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            seed=42,
            stop=["}"],
        )
        return result["choices"][0]["text"]

    @staticmethod
    def _parse_output(text: str) -> dict[str, Any]:
        """Extract JSON from LLM output."""
        # Try to find JSON object in output
        try:
            # Find first '{' and last '}'
            start = text.index("{")
            end = text.rindex("}") + 1
            raw_json = text[start:end]
            parsed = json.loads(raw_json)
            return {"semantic": _flatten_tags(parsed)}
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("llm_json_parse_failed", extra={"output_preview": text[:200], "error": str(exc)})
            return {"semantic": [], "json_error": str(exc)}


def _flatten_tags(parsed: dict[str, Any]) -> list[str]:
    """Flatten {domaine: "financier", priorite: null} → ["domaine:financier", ...]."""
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
