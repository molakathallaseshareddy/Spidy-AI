import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
        if timeout <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")

        model = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
        if not model:
            raise ValueError("OLLAMA_MODEL must not be empty.")

        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=model,
            ollama_timeout_seconds=timeout,
        )