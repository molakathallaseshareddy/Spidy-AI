from typing import Protocol

import httpx

from app.config import Settings


class LLMError(Exception):
    """Raised when the configured language model cannot complete a request."""


class LLMProvider(Protocol):
    async def generate(self, *, system_prompt: str, user_message: str) -> str:
        """Generate a response from the system and user messages."""


class OllamaProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def generate(self, *, system_prompt: str, user_message: str) -> str:
        request_body = {
            "model": self._settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
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
            content = payload["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama returned an invalid response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama returned an empty response.")
        return content.strip()