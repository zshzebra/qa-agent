from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ResearchOutput:
    answer: Any
    reasoning: str


@dataclass
class NextMessage:
    speak: bool
    expects_reply: bool = False
    message: str = ""


@dataclass
class PlanStatus:
    done: bool
    note: str = ""


@dataclass
class AuditResult:
    approved: bool
    issues: str = ""


@dataclass
class SystemOutput:
    kind: Literal["message", "state", "done", "error"]
    message: str = ""
    expects_reply: bool = False
    final_plan: str = ""
    error: str = ""


@dataclass
class DialogueMessage:
    role: Literal["user", "assistant"]
    content: str

    def for_agent(self) -> str:
        label = "User" if self.role == "user" else "Assistant"
        return f"{label}: {self.content}"


@dataclass
class ConversationState:
    dialogue: tuple[DialogueMessage, ...]
    waiting_for_user: bool
    is_done: bool
    done_note: str
    final_plan: str
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
