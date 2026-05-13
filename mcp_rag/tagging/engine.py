"""Tagging orchestrator — H1 rules + H2 LLM via Ollama with cache and fallback."""

import datetime
import logging
from pathlib import Path
from typing import Any

from mcp_rag.ollama_client import OllamaClient
from mcp_rag.tagging.heuristics import HeuristicTagger
from mcp_rag.tagging.llm_tagger import LLMTagger

logger = logging.getLogger(__name__)


class TaggingEngine:
    """Orchestrates H1 (fast heuristics) and H2 (LLM semantic via Ollama) tagging."""

    def __init__(self, config: Any) -> None:
        self.cfg = config.tagging
        self.h1 = HeuristicTagger(max_rules_bytes=config.security.ragrules_max_bytes)
        self.h2 = LLMTagger(config.tagging) if self.cfg.auto_tag_enabled else None

    async def tag_document(
        self,
        path: Path,
        text_preview: str,
        content_hash: str,
        ollama_client: OllamaClient,
        tag_model: str,
    ) -> dict[str, Any]:
        """
        Generate system (H1) and semantic (H2) tags for a document.
        Returns metadata-ready tag dict.
        """
        # H1: Always run, instant
        h1_tags = self.h1.tag_document(path)

        # H2: Run if enabled and Ollama is available
        semantic_result = {"semantic": [], "model": "", "inferred_at": "", "llm_status": "disabled"}
        if self.h2 and tag_model:
            try:
                h2_raw = await self.h2.tag_document(
                    text_preview=text_preview,
                    file_name=path.name,
                    h1_tags=h1_tags,
                    content_hash=content_hash,
                    ollama_client=ollama_client,
                    tag_model=tag_model,
                )
                semantic_tags = h2_raw.get("semantic", [])
                semantic_result = {
                    "semantic": semantic_tags,
                    "model": tag_model,
                    "inferred_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "llm_status": h2_raw.get("llm_status", "ok"),
                }
            except Exception as exc:
                logger.warning("tagging_h2_failed", extra={"path": str(path), "error": str(exc)})
                semantic_result["llm_status"] = "error"

        return {
            "system": h1_tags,
            **semantic_result,
        }
