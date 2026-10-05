import json
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings


class LLMError(Exception):
    """Raised when the configured language model cannot complete a request."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    raw_message: dict[str, object]


class LLMProvider(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LLMResponse:
        """Exchange a conversation, optionally enabling declared tools."""


class OllamaProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def chat(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LLMResponse:
        request_body = {
            "model": self._settings.ollama_model,
            "stream": False,
            "messages": messages,
            "tools": tools,
        }

        try:
            async with httpx.AsyncClient(
                base_url=self._settings.ollama_base_url,
                timeout=self._settings.ollama_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post("/api/chat", json=request_body)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("The Ollama request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                "Unable to connect to Ollama. Confirm it is running and configured."
            ) from exc

        try:
            payload = response.json()
            raw_message = payload["message"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama returned an invalid response.") from exc

        if not isinstance(raw_message, dict):
            raise LLMError("Ollama returned an invalid response.")

        content = raw_message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMError("Ollama returned an invalid response.")

        raw_tool_calls = raw_message.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise LLMError("Ollama returned invalid tool calls.")

        tool_calls: list[ToolCall] = []
        for raw_call in raw_tool_calls:
            try:
                function = raw_call["function"]
                name = function["name"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise TypeError
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise LLMError("Ollama returned invalid tool calls.") from exc
            tool_calls.append(ToolCall(name=name, arguments=arguments))

        normalized_content = content.strip() if isinstance(content, str) else None
        if not normalized_content and not tool_calls:
            raise LLMError("Ollama returned an empty response.")

        return LLMResponse(
            content=normalized_content,
            tool_calls=tool_calls,
            raw_message=raw_message,
        )