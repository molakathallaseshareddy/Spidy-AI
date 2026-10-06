import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.base import ToolResult

MAX_FILE_BYTES = 1_000_000
MAX_DIRECTORY_ENTRIES = 200


class PathInput(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", min_length=1, max_length=4096)


class FileSystemTools:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.expanduser().resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise ValueError("ASSISTANT_WORKSPACE_ROOT must be a directory.")

    @staticmethod
    def _resolve_inside_workspace(root: Path, requested_path: str) -> Path:
        path = Path(requested_path).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PermissionError("Requested path is outside the assistant workspace.") from exc
        return resolved

    def read_file(self, requested_path: str) -> dict[str, object]:
        path = self._resolve_inside_workspace(self.workspace_root, requested_path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {requested_path}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte read limit.")
        content = path.read_text(encoding="utf-8")
        return {"path": str(path), "content": content, "size_bytes": size}

    def list_directory(self, requested_path: str) -> dict[str, object]:
        path = self._resolve_inside_workspace(self.workspace_root, requested_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {requested_path}")
        entries = sorted(path.iterdir(), key=lambda item: item.name.casefold())
        visible_entries = entries[:MAX_DIRECTORY_ENTRIES]
        return {
            "path": str(path),
            "entries": [
                {"name": entry.name, "is_directory": entry.is_dir()}
                for entry in visible_entries
            ],
            "truncated": len(entries) > len(visible_entries),
        }


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file inside the configured assistant workspace."
    permission_level = "low"
    input_model = PathInput

    def __init__(self, filesystem: FileSystemTools) -> None:
        self._filesystem = filesystem

    async def execute(self, arguments: BaseModel) -> ToolResult:
        try:
            result = await asyncio.to_thread(
                self._filesystem.read_file,
                arguments.path,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, data=result)


class ListDirectoryTool:
    name = "list_directory"
    description = "List actual files and directories inside the configured workspace."
    permission_level = "low"
    input_model = ListDirectoryInput

    def __init__(self, filesystem: FileSystemTools) -> None:
        self._filesystem = filesystem

    async def execute(self, arguments: BaseModel) -> ToolResult:
        try:
            result = await asyncio.to_thread(
                self._filesystem.list_directory,
                arguments.path,
            )
        except (OSError, ValueError) as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, data=result)