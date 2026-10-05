import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.tools.base import ToolResult

ApplicationName = Literal["chrome", "edge", "notepad", "calculator"]


class OpenApplicationInput(BaseModel):
    application: ApplicationName


class ApplicationLauncherTool:
    name = "open_application"
    description = (
        "Launch an installed application by supported name: chrome, edge, notepad, "
        "or calculator. No command arguments are accepted."
    )
    input_model = OpenApplicationInput

    @staticmethod
    def _find_executable(application: ApplicationName) -> str | None:
        executable_names = {
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
        }
        executable_name = executable_names[application]
        from_path = shutil.which(executable_name)
        if from_path:
            return from_path

        if application in {"chrome", "edge"}:
            program_roots = [
                os.environ.get("PROGRAMFILES"),
                os.environ.get("PROGRAMFILES(X86)"),
                os.environ.get("LOCALAPPDATA"),
            ]
            product_path = (
                Path("Google/Chrome/Application/chrome.exe")
                if application == "chrome"
                else Path("Microsoft/Edge/Application/msedge.exe")
            )
            for root in program_roots:
                if root:
                    candidate = Path(root) / product_path
                    if candidate.is_file():
                        return str(candidate)
        elif application in {"notepad", "calculator"}:
            windows_root = os.environ.get("WINDIR")
            if windows_root:
                candidate = Path(windows_root) / "System32" / executable_name
                if candidate.is_file():
                    return str(candidate)
        return None

    async def execute(self, arguments: BaseModel) -> ToolResult:
        application = arguments.application
        executable = self._find_executable(application)
        if executable is None:
            return ToolResult(
                success=False,
                error=f"Supported application is not installed or discoverable: {application}.",
            )

        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
            )
        except OSError as exc:
            return ToolResult(success=False, error=f"Unable to launch {application}: {exc}")
        return ToolResult(
            success=True,
            data={"application": application, "process_id": process.pid},
        )