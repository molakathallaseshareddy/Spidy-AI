import asyncio
import json
from pathlib import Path

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


def test_message_returns_openai_response() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["messages"][-1] == {
            "role": "user",
            "content": "Explain FastAPI briefly.",
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "FastAPI is a Python web framework.",
                        }
                    }
                ]
            },
        )

    app = create_app(
        Settings(openai_api_key="test-key", openai_model="test-model"),
        llm_transport=httpx.MockTransport(handle_request),
    )

    response = asyncio.run(post_message(app, "Explain FastAPI briefly."))

    assert response.status_code == 200
    assert response.json() == {
        "response": "FastAPI is a Python web framework."
    }


def test_openai_connection_failure_is_reported() -> None:
    def fail_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = create_app(
        Settings(openai_api_key="test-key"),
        llm_transport=httpx.MockTransport(fail_request),
    )

    response = asyncio.run(post_message(app, "Hello"))

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Unable to connect to OpenAI. Check your internet connection."
    }


def test_missing_openai_api_key_is_reported() -> None:
    app = create_app(Settings())

    response = asyncio.run(post_message(app, "Hello"))

    assert response.status_code == 503
    assert response.json() == {
        "detail": "OPENAI_API_KEY is not configured. Add your OpenAI API key to .env."
    }


def test_tool_call_reads_real_workspace_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The project uses FastAPI.", encoding="utf-8")
    requests: list[dict[str, object]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert any(
                tool["function"]["name"] == "read_file"
                for tool in body["tools"]
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "refusal": None,
                                "annotations": [],
                                "tool_calls": [
                                    {
                                        "id": "call_read_file",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"notes.txt"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        tool_message = next(
            message for message in body["messages"] if message["role"] == "tool"
        )
        assert tool_message["tool_call_id"] == "call_read_file"
        tool_result = json.loads(tool_message["content"])
        assert tool_result["success"] is True
        assert tool_result["data"]["content"] == "The project uses FastAPI."
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "It uses FastAPI.",
                        }
                    }
                ]
            },
        )

    app = create_app(
        Settings(
            openai_api_key="test-key",
            openai_model="test-model",
            workspace_root=tmp_path,
        ),
        llm_transport=httpx.MockTransport(handle_request),
    )
    response = asyncio.run(post_message(app, "What framework is this project using?"))

    assert response.status_code == 200
    assert response.json() == {"response": "It uses FastAPI."}
    assert len(requests) == 2