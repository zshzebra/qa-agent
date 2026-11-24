from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.common_tools.exa import ExaToolset

type SourcesCallback = Callable[[Any, list[str]], Awaitable[None]]
"""Called with the run's deps and the domain of each source a result cited."""


class ExaWithoutCitation:
    def __init__(self, exa_api_key: str, on_sources: SourcesCallback) -> None:
        self.on_sources = on_sources
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
        """Report the domains a result cited, then drop `exa_answer` citations."""
        domains = source_domains(result)
        if domains:
            await self.on_sources(ctx.deps, domains)
        # TODO: Don't drop the citations
        if isinstance(result, dict) and "citations" in result:
            result.pop("citations", None)
        return result


def source_domains(result: Any) -> list[str]:
    """The domain of each source in an `exa_answer` or `exa_find_similar` result."""
    if isinstance(result, dict):
        pages = result.get("citations") or []
    elif isinstance(result, list):
        pages = result
    else:
        return []
    domains: list[str] = []
    for page in pages:
        url = page.get("url") if isinstance(page, dict) else None
        domain = domain_of(url) if isinstance(url, str) else None
        if domain:
            domains.append(domain)
    return domains


def domain_of(url: str) -> str | None:
    host = urlsplit(url).hostname
    if not host:
        return None
    return host.removeprefix("www.")
