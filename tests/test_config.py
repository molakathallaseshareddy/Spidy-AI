from pathlib import Path

import app.config as config_module
from app.config import Settings


def test_settings_load_from_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'OLLAMA_BASE_URL = "http://ollama.test:11434"\n'
        'OLLAMA_MODEL = "llama3.1:latest"\n'
        "OLLAMA_TIMEOUT_SECONDS = 180\n"
        f'ASSISTANT_WORKSPACE_ROOT = "{(tmp_path / "workspace root").as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_ENV_FILE", env_file)
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT_SECONDS",
        "ASSISTANT_WORKSPACE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.ollama_base_url == "http://ollama.test:11434"
    assert settings.ollama_model == "llama3.1:latest"
    assert settings.ollama_timeout_seconds == 180
    assert settings.workspace_root == (tmp_path / "workspace root").resolve()


def test_process_environment_overrides_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "_ENV_FILE", env_file)
    monkeypatch.setenv("OLLAMA_MODEL", "from-process")

    assert Settings.from_env().ollama_model == "from-process"
