import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 60.0
    workspace_root: Path = Path.cwd()

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(_ENV_FILE, override=False)

        timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
        if timeout <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than zero.")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        if not model:
            raise ValueError("OPENAI_MODEL must not be empty.")

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=model,
            openai_timeout_seconds=timeout,
            workspace_root=Path(
                os.getenv("ASSISTANT_WORKSPACE_ROOT", str(Path.cwd()))
            ).expanduser().resolve(),
        )