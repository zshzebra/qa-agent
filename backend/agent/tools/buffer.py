from typing import Protocol, runtime_checkable

from pydantic_ai import FunctionToolset, ModelRetry, RunContext


@runtime_checkable
class HasBuffer(Protocol):
    buffer: str


class BufferToolset(FunctionToolset[HasBuffer]):
    def __init__(self, *, context_lines: int = 2) -> None:
        super().__init__()
        self.context_lines = context_lines

        self.add_function(self.read_buffer)
        self.add_function(self.write_buffer)
        self.add_function(self.edit_buffer)
        self.add_function(self.clear_buffer)

        @self.instructions
        def _(ctx: RunContext[HasBuffer]) -> str:
            return (
                "You have CRUD access to a text buffer."
                "Read buffer returns the contents, optionally between a range, with line numbers."
                "Write buffer will set the contents of the buffer. The buffer needs to be empty, orelse it will fail."
                "Edit buffer accepts a string to replace, and it's value. The target must match exactly once across the whole file."
                "Clear buffer nulls the contents."
            )

    def read_buffer(
        self, ctx: RunContext[HasBuffer], start: int = 1, end: int | None = None
    ) -> str:
        lines = ctx.deps.buffer.split("\n")
        end = end or len(lines)
        return "\n".join(f"{i}| {lines[i - 1]}" for i in range(start, end + 1))

    def write_buffer(self, ctx: RunContext[HasBuffer], content: str) -> str:
        if ctx.deps.buffer:
            raise ModelRetry("buffer not empty")
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
