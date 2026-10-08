import json
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings


class LLMError(Exception):
    """Raised when the configured language model cannot complete a request."""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
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


class OpenAIProvider:
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
        if not self._settings.openai_api_key:
            raise LLMError(
                "OPENAI_API_KEY is not configured. Add your OpenAI API key to .env."
            )

        request_body = {
            "model": self._settings.openai_model,
            "stream": False,
            "messages": messages,
            "tools": tools,
        }

        try:
            async with httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self._settings.openai_api_key}"
                },
                timeout=self._settings.openai_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post("/chat/completions", json=request_body)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("The OpenAI request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"OpenAI returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                "Unable to connect to OpenAI. Check your internet connection."
            ) from exc

        try:
            payload = response.json()
            raw_message = payload["choices"][0]["message"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("OpenAI returned an invalid response.") from exc

        if not isinstance(raw_message, dict):
            raise LLMError("OpenAI returned an invalid response.")

        content = raw_message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMError("OpenAI returned an invalid response.")

        raw_tool_calls = raw_message.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise LLMError("OpenAI returned invalid tool calls.")

        tool_calls: list[ToolCall] = []
        normalized_tool_calls: list[dict[str, object]] = []
        for raw_call in raw_tool_calls:
            try:
                call_id = raw_call["id"]
                function = raw_call["function"]
                name = function["name"]
                arguments = function["arguments"]
                if (
                    not isinstance(call_id, str)
                    or not isinstance(name, str)
                    or not isinstance(arguments, str)
                ):
                    raise TypeError
                parsed_arguments = json.loads(arguments)
                if not isinstance(parsed_arguments, dict):
                    raise TypeError
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise LLMError("OpenAI returned invalid tool calls.") from exc
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=parsed_arguments,
                )
            )
            normalized_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )

        normalized_content = content.strip() if isinstance(content, str) else None
        if not normalized_content and not tool_calls:
            raise LLMError("OpenAI returned an empty response.")

        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": content,
        }
        if normalized_tool_calls:
            assistant_message["tool_calls"] = normalized_tool_calls

        return LLMResponse(
            content=normalized_content,
            tool_calls=tool_calls,
            raw_message=assistant_message,
        )