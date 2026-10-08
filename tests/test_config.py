from pathlib import Path

import app.config as config_module
from app.config import Settings


def test_settings_load_from_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'OPENAI_API_KEY = "test-key"\n'
        'OPENAI_MODEL = "gpt-4o-mini"\n'
        "OPENAI_TIMEOUT_SECONDS = 180\n"
        f'ASSISTANT_WORKSPACE_ROOT = "{(tmp_path / "workspace root").as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_ENV_FILE", env_file)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_TIMEOUT_SECONDS",
        "ASSISTANT_WORKSPACE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_timeout_seconds == 180
    assert settings.workspace_root == (tmp_path / "workspace root").resolve()


def test_process_environment_overrides_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "_ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_MODEL", "from-process")

    assert Settings.from_env().openai_model == "from-process"
