from pathlib import Path

import pytest

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