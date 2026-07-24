from typing import Any

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.common_tools.exa import ExaToolset


class ExaWithoutCitation:
    def __init__(self, exa_api_key: str) -> None:
        self.exa_hooks = Hooks(after_tool_execute=self.process_tool_result)
        self.exa_toolset = ExaToolset(
            exa_api_key,
            num_results=5,
            max_characters=1000,
            include_search=False,
            include_find_similar=True,
            include_get_contents=False,
            include_answer=True,
        )

    async def process_tool_result(
        self, ctx, /, *, call, tool_def: ToolDefinition, args, result: Any
    ):
        """Tool hook to drop citations from `exa_answer` results."""
        # TODO: Don't drop the citations
        if "citations" in result:
            result.pop("citations", None)
        return result
