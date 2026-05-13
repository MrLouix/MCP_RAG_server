"""Tests for OllamaClient — mocked HTTP, no real Ollama needed."""

import asyncio
import json

import httpx
import pytest

from mcp_rag.ollama_client import OllamaClient, image_to_base64


@pytest.fixture
def mock_transport():
    """Build an httpx MockTransport that fakes Ollama endpoints."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}

        if path == "/api/embed":
            inputs = body.get("input", [])
            # Return dummy 4-dim vectors
            embeddings = [[0.1, 0.2, 0.3, 0.4]] * len(inputs)
            return httpx.Response(200, json={"embeddings": embeddings})

        if path == "/api/chat":
            if body.get("keep_alive") == 0:
                # unload request
                return httpx.Response(200, json={"message": {"content": ""}})
            messages = body.get("messages", [])
            content = messages[0].get("content", "") if messages else ""
            if body.get("format") == "json":
                resp = json.dumps({"domaine": "technique", "priorite": "normal"})
            elif messages and messages[0].get("images"):
                resp = "Extracted text from image"
            else:
                resp = "ok"
            return httpx.Response(200, json={"message": {"content": resp}})

        if path == "/api/tags":
            return httpx.Response(200, json={
                "models": [
                    {"name": "nomic-embed-text", "size": 274000000},
                    {"name": "qwen2.5:3b", "size": 1900000000},
                ]
            })

        if path == "/api/ps":
            return httpx.Response(200, json={
                "models": [{"name": "nomic-embed-text", "size": 274000000}]
            })

        if path == "/api/pull":
            return httpx.Response(200, json={"status": "success"})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.fixture
def client(mock_transport):
    c = OllamaClient(base_url="http://fake-ollama:11434", timeout_s=5.0)
    c._client = httpx.AsyncClient(transport=mock_transport, base_url="http://fake-ollama:11434")
    return c


@pytest.mark.asyncio
async def test_embed(client):
    vectors = await client.embed(["hello", "world"], model="nomic-embed-text")
    assert len(vectors) == 2
    assert len(vectors[0]) == 4


@pytest.mark.asyncio
async def test_embed_batching(client):
    texts = [f"text_{i}" for i in range(70)]
    vectors = await client.embed(texts, model="nomic-embed-text", batch_size=32)
    assert len(vectors) == 70


@pytest.mark.asyncio
async def test_chat_json(client):
    result = await client.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": "classify this"}],
        format="json",
    )
    parsed = json.loads(result)
    assert parsed["domaine"] == "technique"


@pytest.mark.asyncio
async def test_chat_vision(client):
    result = await client.chat_vision(
        model="minicpm-v",
        prompt="Extract text",
        images=["base64data"],
    )
    assert "Extracted text" in result


@pytest.mark.asyncio
async def test_list_models(client):
    models = await client.list_models()
    names = [m["name"] for m in models]
    assert "nomic-embed-text" in names
    assert "qwen2.5:3b" in names


@pytest.mark.asyncio
async def test_list_running(client):
    running = await client.list_running()
    assert len(running) == 1
    assert running[0]["name"] == "nomic-embed-text"


@pytest.mark.asyncio
async def test_healthcheck(client):
    result = await client.healthcheck(required_models=["nomic-embed-text", "qwen2.5:3b", "missing-model"])
    assert result["reachable"] is True
    assert "missing-model" in result["missing_models"]
    assert "nomic-embed-text" not in result["missing_models"]


@pytest.mark.asyncio
async def test_preload(client):
    # Should not raise
    await client.preload("nomic-embed-text")


@pytest.mark.asyncio
async def test_unload(client):
    # Should not raise
    await client.unload("qwen2.5:3b")


@pytest.mark.asyncio
async def test_pull_model(client):
    result = await client.pull_model("some-model")
    assert result["status"] == "success"


def test_image_to_base64(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    b64 = image_to_base64(img)
    assert isinstance(b64, str)
    assert len(b64) > 0
