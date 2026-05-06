"""Embedding wrapper ONNX Runtime — zero PyTorch.

Uses onnxruntime + tokenizers to run sentence-transformers models.
Downloads from HuggingFace Hub using huggingface_hub.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession # type: ignore
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Default model — ONNX-converted version available on HF Hub
# Using Xenova's conversion which provides tokenizer.json + onnx/model.onnx
DEFAULT_MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """Embeds texts using ONNX Runtime — no PyTorch, no CUDA."""

    def __init__(self, model_manager) -> None:
        self._mm = model_manager
        self._tokenizer: Tokenizer | None = None
        self._session: InferenceSession | None = None
    async def _ensure_loaded(self) -> None:
        """Load tokenizer + ONNX model from local volume (/app/models)."""
        if self._session is not None:
            return

        import os
        import shutil
        from huggingface_hub import hf_hub_download

        model_name = self._mm.rag_cfg.embedding.model or DEFAULT_MODEL
        # Safe dir name from model name
        safe_name = model_name.replace("/", "_")
        model_dir = Path("/app/models") / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Helper: download if not exists locally
        def _download(filename: str, dest: Path) -> None:
            if dest.exists():
                return
            local = Path(hf_hub_download(repo_id=model_name, filename=filename))
            shutil.copy2(local, dest)

        tok_path = model_dir / "tokenizer.json"
        onnx_path = model_dir / "model.onnx"

        # Download on first run only (requires network once)
        _download("tokenizer.json", tok_path)
        if not onnx_path.exists():
            for fname in ("onnx/model.onnx", "onnx/model_quantized.onnx", "model.onnx"):
                try:
                    remote = hf_hub_download(repo_id=model_name, filename=fname)
                    shutil.copy2(remote, onnx_path)
                    break
                except Exception:
                    continue

        if not onnx_path.exists():
            raise RuntimeError(
                f"No ONNX model found in {model_name}. "
                "Run once with network access, then the model is cached locally."
            )

        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._session = InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        assert self._session is not None

        # Detect ONNX input names from graph
        input_names = [inp.name for inp in self._session.get_inputs()]
        self._input_ids_name = next((n for n in input_names if "input_ids" in n), input_names[0])
        self._attn_mask_name = next((n for n in input_names if "attention" in n or "mask" in n), None)
        self._has_token_type = any("token_type" in n for n in input_names)

        self._model_name = model_name
        logger.info("onnx_embedder_loaded", extra={
            "model": model_name,
            "onnx_path": str(onnx_path),
        })

    async def embed(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into dense L2-normalized vectors."""
        await self._ensure_loaded()

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            assert self._tokenizer is not None
            encoded = self._tokenizer.encode_batch(
                [t[:512] for t in batch],
                add_special_tokens=True,
            )

            # Manual padding (tokenizers library doesn't support padding in encode_batch)
            max_len = max(len(e.ids) for e in encoded)
            input_ids = []
            attention_masks = []
            for e in encoded:
                n_pad = max_len - len(e.ids)
                input_ids.append(e.ids + [0] * n_pad)
                attention_masks.append(e.attention_mask + [0] * n_pad)

            # ONNX inference
            ort_inputs = {
                self._input_ids_name: np.array(input_ids, dtype=np.int64),
            }
            if self._attn_mask_name:
                ort_inputs[self._attn_mask_name] = np.array(attention_masks, dtype=np.int64)
            if self._has_token_type:
                ort_inputs["token_type_ids"] = np.zeros_like(np.array(input_ids, dtype=np.int64))

            assert self._session is not None
            outputs = self._session.run(None, ort_inputs)
            # outputs[0] = last_hidden_state (batch, seq_len, hidden)
            last_hidden = outputs[0]

            # Mean pooling with attention mask
            mask = np.array(attention_masks)[:, :, np.newaxis]
            sum_embeddings = np.sum(last_hidden * mask, axis=1)
            sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask

            # L2 normalization
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norms, a_min=1e-9, a_max=None)

            all_embeddings.append(normalized)

        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.vstack(all_embeddings)

    async def get_model_name(self) -> str:
        """Return the resolved model name (triggers load)."""
        await self._ensure_loaded()
        return self._model_name or "onnx-embedder"
