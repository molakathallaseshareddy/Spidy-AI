import asyncio
from pathlib import Path

import pytest

import app.tools.browser as browser_module
from app.tools import build_tool_registry
from app.tools.browser import BrowserSearchInput, BrowserSearchTool
from app.tools.filesystem import FileSystemTools


def test_filesystem_reads_actual_text_file(tmp_path: Path) -> None:
    target = tmp_path / "actual.txt"
    target.write_text("real contents", encoding="utf-8")
    filesystem = FileSystemTools(tmp_path)

    result = filesystem.read_file("actual.txt")

    assert result["content"] == "real contents"
    assert result["size_bytes"] == len("real contents")


def test_filesystem_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    filesystem = FileSystemTools(workspace)

    with pytest.raises(PermissionError, match="outside the assistant workspace"):
        filesystem.read_file("../outside.txt")


def test_filesystem_reports_missing_file(tmp_path: Path) -> None:
    filesystem = FileSystemTools(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        filesystem.read_file("missing.txt")


def test_browser_search_returns_actual_page_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class Page:
        url = "https://html.duckduckgo.com/html/?q=FastAPI"

        async def goto(self, url: str, **kwargs: object) -> None:
            assert url == self.url
            assert kwargs["wait_until"] == "domcontentloaded"

        def locator(self, selector: str) -> "Page":
            assert selector == "body"
            return self

        async def inner_text(self) -> str:
            return "Actual search results"

        async def title(self) -> str:
            return "Search results"

    class Browser:
        async def new_page(self) -> Page:
            return Page()

        async def close(self) -> None:
            pass

    class Chromium:
        async def launch(self) -> Browser:
            return Browser()

    class PlaywrightContext:
        def __init__(self) -> None:
            self.chromium = Chromium()

        async def __aenter__(self) -> "PlaywrightContext":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        browser_module,
        "async_playwright",
        lambda: PlaywrightContext(),
    )

    result = asyncio.run(
        BrowserSearchTool().execute(BrowserSearchInput(query="FastAPI"))
    )

    assert result.success is True
    assert result.data == {
        "query": "FastAPI",
        "url": "https://html.duckduckgo.com/html/?q=FastAPI",
        "title": "Search results",
        "content": "Actual search results",
        "truncated": False,
    }


def test_browser_search_tool_is_registered() -> None:
    registry = build_tool_registry(Path.cwd())

    assert "browser_search" in {
        tool["function"]["name"] for tool in registry.definitions
    }


def test_browser_search_reports_missing_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Chromium:
        async def launch(self) -> None:
            raise browser_module.PlaywrightError("Chromium is not installed")

    class PlaywrightContext:
        chromium = Chromium()

        async def __aenter__(self) -> "PlaywrightContext":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        browser_module,
        "async_playwright",
        lambda: PlaywrightContext(),
    )

    result = asyncio.run(
        BrowserSearchTool().execute(BrowserSearchInput(query="FastAPI"))
    )

    assert result.success is False
    assert result.error == (
        "Unable to complete web search: Chromium is not installed"
    )