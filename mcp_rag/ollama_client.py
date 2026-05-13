"""Async HTTP client for an external Ollama instance.

Handles embedding, chat (JSON mode for tagging), and vision (OCR) endpoints.
All ML inference is delegated to Ollama — zero local models.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Sequence

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client wrapping the Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://172.28.128.1:11434",
        timeout_s: float = 30.0,
        embed_timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.embed_timeout_s = embed_timeout_s
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Embedding  — POST /api/embed
    # ------------------------------------------------------------------

    async def embed(
        self,
        texts: Sequence[str],
        model: str,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed texts via Ollama. Batches internally to avoid oversized payloads."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            payload = {"model": model, "input": batch}
            data = await self._post("/api/embed", payload, timeout=self.embed_timeout_s)
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(batch)} inputs"
                )
            all_vectors.extend(embeddings)
        return all_vectors

    # ------------------------------------------------------------------
    # Chat  — POST /api/chat  (JSON mode for tagging)
    # ------------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        format: str = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request. Returns the assistant message content."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Vision / OCR  — POST /api/chat with images
    # ------------------------------------------------------------------

    async def chat_vision(
        self,
        model: str,
        prompt: str,
        images: list[str],
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a multimodal chat request with base64-encoded images."""
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": images,
            }
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Model management helpers
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict[str, Any]]:
        """GET /api/tags — list locally available models on the Ollama instance."""
        data = await self._get("/api/tags")
        return data.get("models", [])

    async def list_running(self) -> list[dict[str, Any]]:
        """GET /api/ps — list models currently loaded in memory on Ollama."""
        data = await self._get("/api/ps")
        return data.get("models", [])

    async def preload(self, model: str) -> None:
        """Warm-up a model by sending a minimal request."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1},
        }
        await self._post("/api/chat", payload)

    async def unload(self, model: str) -> None:
        """Ask Ollama to unload a model from memory (keep_alive=0)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "keep_alive": 0,
            "options": {"num_predict": 1},
        }
        try:
            await self._post("/api/chat", payload)
        except Exception as exc:
            logger.warning("ollama_unload_failed", extra={"model": model, "error": str(exc)})

    async def pull_model(self, model: str) -> dict[str, Any]:
        """POST /api/pull — download a model on the Ollama instance."""
        return await self._post("/api/pull", {"name": model, "stream": False})

    async def healthcheck(self, required_models: list[str] | None = None) -> dict[str, Any]:
        """Check Ollama connectivity and model availability."""
        result: dict[str, Any] = {
            "reachable": False,
            "base_url": self.base_url,
            "models_available": [],
            "models_running": [],
            "missing_models": [],
        }
        try:
            models = await self.list_models()
            result["reachable"] = True
            available_names = [m.get("name", "").split(":")[0] for m in models]
            # Also keep full names (with tags)
            available_full = [m.get("name", "") for m in models]
            result["models_available"] = available_full

            running = await self.list_running()
            result["models_running"] = [m.get("name", "") for m in running]

            if required_models:
                for req in required_models:
                    req_base = req.split(":")[0]
                    if req not in available_full and req_base not in available_names:
                        result["missing_models"].append(req)
        except Exception as exc:
            logger.warning("ollama_healthcheck_failed", extra={"error": str(exc)})

        return result

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST with retry + backoff."""
        client = await self._get_client()
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = await client.post(
                    path,
                    json=payload,
                    timeout=timeout or self.timeout_s,
                )
                r.raise_for_status()
                return r.json()
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        "ollama_request_retry",
                        extra={"path": path, "attempt": attempt, "error": str(exc), "wait_s": wait},
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Ollama request failed after {self.max_retries} attempts: {last_exc}")

    async def _get(self, path: str) -> dict[str, Any]:
        """Simple GET with retry."""
        client = await self._get_client()
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = await client.get(path, timeout=self.timeout_s)
                r.raise_for_status()
                return r.json()
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2 ** (attempt - 1)
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Ollama GET {path} failed after {self.max_retries} attempts: {last_exc}")


def image_to_base64(path: Path) -> str:
    """Read an image file and return its base64 encoding."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")
