"""Embedding wrapper around sentence-transformers with lazy loading via ModelManager."""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper that delegates model loading to ModelManager and exposes embed()."""

    def __init__(self, model_manager) -> None:
        self._mm = model_manager
        self._model_name: str | None = None

    async def embed(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into dense vectors. Returns ndarray of shape (N, dim)."""
        model = await self._mm.get_embedder()
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if self._model_name is None:
            # Stocke le nom réel utilisé pour les métadonnées
            self._model_name = getattr(
                model,
                "model_name",
                model.get_sentence_embedding_dimension(),
            )
        return embeddings

    async def get_model_name(self) -> str:
        """Return the resolved model name (triggers load)."""
        _ = await self._mm.get_embedder()
        if self._model_name is None:
            self._model_name = "sentence-transformers"
        return self._model_name
