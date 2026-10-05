from pydantic import BaseModel, ValidationError

from app.llm import ToolCall
from app.tools.applications import ApplicationLauncherTool
from app.tools.base import AssistantTool
from app.tools.filesystem import FileSystemTools, ListDirectoryTool, ReadFileTool


class ToolRegistry:
    def __init__(self, tools: list[AssistantTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique.")

    @property
    def definitions(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, call: ToolCall) -> dict[str, object]:
        tool = self._tools.get(call.name)
        if tool is None:
            return {"success": False, "error": f"Unknown tool: {call.name}."}

        try:
            arguments: BaseModel = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return {"success": False, "error": f"Invalid tool input: {exc}"}

        try:
            return (await tool.execute(arguments)).to_dict()
        except Exception:
            return {"success": False, "error": "Tool execution failed unexpectedly."}


def build_tool_registry(workspace_root) -> ToolRegistry:
    filesystem = FileSystemTools(workspace_root)
    return ToolRegistry(
        [
            ListDirectoryTool(filesystem),
            ReadFileTool(filesystem),
            ApplicationLauncherTool(),
        ]
    )