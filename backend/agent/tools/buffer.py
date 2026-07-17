from typing import Protocol, runtime_checkable

from pydantic_ai import FunctionToolset, ModelRetry, RunContext


@runtime_checkable
class HasBuffer(Protocol):
    buffer: str


class BufferToolset(FunctionToolset[HasBuffer]):
    def __init__(self, *, context_lines: int = 2) -> None:
        super().__init__()
        self.context_lines = context_lines

        self.add_function(self.read_buffer, retries=3)
        self.add_function(self.write_buffer, retries=3)
        self.add_function(self.edit_buffer, retries=3)
        self.add_function(self.clear_buffer, retries=3)

        @self.instructions
        def _(ctx: RunContext[HasBuffer]) -> str:
            return (
                "You have CRUD access to a single text buffer.\n"
                "- read_buffer returns the contents (optionally a line range) with "
                "line numbers.\n"
                "- write_buffer sets the whole buffer, but only when it is empty; "
                "if it already has content it will fail, so use edit_buffer "
                "instead.\n"
                "- edit_buffer replaces an exact substring with a new value — the "
                "target must appear exactly once, so include enough surrounding "
                "text to make it unique, otherwise the edit is rejected.\n"
                "- clear_buffer empties the buffer.\n"
                "Prefer edit_buffer for incremental changes; use write_buffer only "
                "for the initial fill."
            )

    def read_buffer(
        self, ctx: RunContext[HasBuffer], start: int = 1, end: int | None = None
    ) -> str:
        lines = ctx.deps.buffer.split("\n")
        if end is None:
            end = len(lines)
        if start < 1 or end > len(lines) or start > end:
            raise ModelRetry(
                f"range {start}-{end} is out of bounds; buffer has {len(lines)} "
                f"line(s), valid range is 1-{len(lines)}"
            )
        return "\n".join(f"{i}| {lines[i - 1]}" for i in range(start, end + 1))

    def write_buffer(self, ctx: RunContext[HasBuffer], content: str) -> str:
        if ctx.deps.buffer:
            raise ModelRetry(
                "buffer not empty. Either use `edit_buffer` or clear the the buffer first, before writing to it."
            )
        ctx.deps.buffer = content
        return content

    def edit_buffer(
        self, ctx: RunContext[HasBuffer], target: str, replacement: str
    ) -> str:
        buffer = ctx.deps.buffer
        count = buffer.count(target)
        if count != 1:
            raise ModelRetry(f"target matches {count} times, need exactly 1")

        before = buffer[: buffer.index(target)]
        first = before.count("\n") + 1
        last = first + replacement.count("\n")

        buffer = buffer.replace(target, replacement)
        ctx.deps.buffer = buffer

        lines = buffer.split("\n")
        lo = max(1, first - self.context_lines)
        hi = min(len(lines), last + self.context_lines)
        return "\n".join(f"{i}| {lines[i - 1]}" for i in range(lo, hi + 1))

    def clear_buffer(self, ctx: RunContext[HasBuffer]) -> None:
        ctx.deps.buffer = ""
