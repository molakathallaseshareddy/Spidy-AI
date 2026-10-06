from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolResult:
    success: bool
    data: dict[str, object] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result


class AssistantTool(Protocol):
    name: str
    description: str
    permission_level: str
    input_model: type[BaseModel]

    async def execute(self, arguments: BaseModel) -> ToolResult:
        """Execute validated arguments and return the actual operation result."""