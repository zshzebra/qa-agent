from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass
class ResearchOutput:
    answer: Any
    reasoning: str


@dataclass
class NextMessage:
    speak: bool
    expects_reply: bool = False
    message: str = ""


class ContinuePlanning(BaseModel):
    """End this planner cycle without completing the collection."""


class AuditIssue(BaseModel):
    field: str | None = Field(
        description=(
            "The relevant target field, using a dotted path for a nested field, "
            "or null for a collection-wide issue."
        )
    )
    reason: str = Field(
        description=(
            "Why the submitted value is unsupported, unconfirmed, contradictory, "
            "or otherwise not ready."
        )
    )


class AuditResult(BaseModel):
    issues: list[AuditIssue] = Field(
        description="Empty only when the submitted candidate is fully approved.",
    )


@dataclass
class SystemOutput[OutputT: BaseModel]:
    kind: Literal["message", "state", "done", "error"]
    message: str = ""
    expects_reply: bool = False
    result: OutputT | None = None
    error: str = ""


@dataclass
class DialogueMessage:
    role: Literal["user", "assistant"]
    content: str

    def for_agent(self) -> str:
        label = "User" if self.role == "user" else "Assistant"
        return f"{label}: {self.content}"


@dataclass
class ConversationState[OutputT: BaseModel]:
    dialogue: tuple[DialogueMessage, ...]
    waiting_for_user: bool
    is_done: bool
    done_note: str
    result: OutputT | None
    failure: str | None


def render_dialogue(messages: list[DialogueMessage]) -> str:
    return "\n".join(message.for_agent() for message in messages)


@dataclass
class ResearchJob:
    id: str
    target: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    result: str | None = None
    error: str | None = None
    completion_order: int | None = None
    consumed: bool = False
