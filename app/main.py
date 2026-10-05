import logging
import time
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import Assistant
from app.config import Settings
from app.llm import LLMError, OllamaProvider

logger = logging.getLogger(__name__)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000)


class MessageResponse(BaseModel):
    response: str


def create_app(
    settings: Settings | None = None,
    *,
    ollama_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app = FastAPI(title="AI Personal Assistant", version="0.1.0")
    configuration = settings or Settings.from_env()
    app.state.assistant = Assistant(
        OllamaProvider(configuration, transport=ollama_transport)
    )

    @app.post("/api/assistant/message", response_model=MessageResponse)
    async def send_message(request: MessageRequest) -> MessageResponse:
        request_id = uuid4().hex
        started_at = time.perf_counter()
        try:
            response = await app.state.assistant.respond(request.message)
        except LLMError as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "assistant request failed request_id=%s duration_ms=%.1f error=%s",
                request_id,
                duration_ms,
                exc,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "assistant request completed request_id=%s duration_ms=%.1f",
            request_id,
            duration_ms,
        )
        return MessageResponse(response=response)

    return app


app = create_app()