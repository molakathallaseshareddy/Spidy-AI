import asyncio
import json

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.main import create_app


async def post_message(app: FastAPI, message: str) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.post(
            "/api/assistant/message",
            json={"message": message},
        )


def test_message_returns_ollama_response() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["messages"][-1] == {
            "role": "user",
            "content": "Explain FastAPI briefly.",
        }
        return httpx.Response(
            200,
            json={"message": {"content": "FastAPI is a Python web framework."}},
        )

    app = create_app(
        Settings(ollama_model="test-model"),
        ollama_transport=httpx.MockTransport(handle_request),
    )

    response = asyncio.run(post_message(app, "Explain FastAPI briefly."))

    assert response.status_code == 200
    assert response.json() == {
        "response": "FastAPI is a Python web framework."
    }


def test_ollama_connection_failure_is_reported() -> None:
    def fail_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = create_app(
        Settings(),
        ollama_transport=httpx.MockTransport(fail_request),
    )

    response = asyncio.run(post_message(app, "Hello"))

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Unable to connect to Ollama. Confirm it is running and configured."
    }