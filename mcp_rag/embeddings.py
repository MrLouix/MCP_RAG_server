"""Embedding wrapper via Ollama /api/embed — zero local models."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from mcp_rag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class Embedder:
    """Embeds texts by calling an external Ollama instance."""

    def __init__(self, ollama: OllamaClient, model: str) -> None:
        self._ollama = ollama
        self._model = model

    async def embed(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts into dense L2-normalized vectors via Ollama."""
        vectors = await self._ollama.embed(
            texts=list(texts),
            model=self._model,
            batch_size=batch_size,
        )
        arr = np.array(vectors, dtype=np.float32)
        # L2 normalization
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, a_min=1e-9, a_max=None)

    async def get_model_name(self) -> str:
        return self._model
