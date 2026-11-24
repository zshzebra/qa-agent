from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Annotated, AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel

from onboarding import ExampleCompanyOnboarding

from .system import AgentSystem


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ActiveResearcher(BaseModel):
    id: str
    domains: list[str]


class SessionSnapshot[OutputT: BaseModel](BaseModel):
    revision: int
    status: Literal["working", "waiting_for_user", "completed", "failed"]
    messages: list[SessionMessage]
    completion_message: str | None = None
    result: OutputT | None = None
    error: str | None = None
    researchers: list[ActiveResearcher] = []


class SendMessageRequest(BaseModel):
    content: str


class SessionSupervisor[OutputT: BaseModel]:
    def __init__(
        self,
        system: AgentSystem[OutputT],
    ) -> None:
        self.system = system
        self._start_lock = asyncio.Lock()
        self._publish_lock = asyncio.Lock()
        self._output_task: asyncio.Task[None] | None = None
        self._subscribers: set[asyncio.Queue[SessionSnapshot[OutputT]]] = set()
        self._revision = 0

    async def start(self) -> None:
        async with self._start_lock:
            if self._output_task is not None:
                return
            await self.system.run()
            self._output_task = asyncio.create_task(self._pump_outputs())

    async def snapshot(self) -> SessionSnapshot[OutputT]:
        async with self._publish_lock:
            return await self._snapshot()

    async def send(self, content: str) -> SessionSnapshot[OutputT]:
        await self.system.send(content)
        return await self._publish()

    async def subscribe(self) -> asyncio.Queue[SessionSnapshot[OutputT]]:
        subscriber: asyncio.Queue[SessionSnapshot[OutputT]] = asyncio.Queue(
            maxsize=1
        )
        async with self._publish_lock:
            self._subscribers.add(subscriber)
            subscriber.put_nowait(await self._snapshot())
        return subscriber

    async def unsubscribe(
        self,
        subscriber: asyncio.Queue[SessionSnapshot[OutputT]],
    ) -> None:
        async with self._publish_lock:
            self._subscribers.discard(subscriber)

    async def close(self) -> None:
        if self._output_task is not None:
            self._output_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._output_task
            self._output_task = None
        await self.system.close()

    async def _pump_outputs(self) -> None:
        while True:
            await self.system.receive()
            await self._publish()

    async def _publish(self) -> SessionSnapshot[OutputT]:
        async with self._publish_lock:
            self._revision += 1
            snapshot = await self._snapshot()
            for subscriber in self._subscribers:
                if subscriber.full():
                    subscriber.get_nowait()
                subscriber.put_nowait(snapshot)
            return snapshot

    async def _snapshot(self) -> SessionSnapshot[OutputT]:
        state = await self.system.conversation_state()
        if state.failure is not None:
            session_status = "failed"
        elif state.is_done:
            session_status = "completed"
        elif state.waiting_for_user:
            session_status = "waiting_for_user"
        else:
            session_status = "working"

        return SessionSnapshot(
            revision=self._revision,
            status=session_status,
            messages=[
                SessionMessage(role=message.role, content=message.content)
                for message in state.dialogue
            ],
            completion_message=state.done_note or None,
            result=state.result,
            error=state.failure,
            researchers=[
                ActiveResearcher(id=r.id, domains=list(r.domains))
                for r in state.researchers
            ],
        )


AppSupervisor = SessionSupervisor[ExampleCompanyOnboarding]
AppSnapshot = SessionSnapshot[ExampleCompanyOnboarding]


class SessionRegistry:
    def __init__(self) -> None:
        self._supervisors: dict[str, AppSupervisor] = {}
        self._lock = asyncio.Lock()
        self._model = GoogleModel("gemini-3.1-flash-lite-preview")

    async def get(
        self,
        key: str,
    ) -> AppSupervisor:
        async with self._lock:
            supervisor = self._supervisors.get(key)
            if supervisor is None:
                supervisor = SessionSupervisor(
                    AgentSystem(
                        output_type=ExampleCompanyOnboarding,
                        planner_model=self._model,
                        speaker_model=self._model,
                        researcher_model=self._model,
                        auditor_model=self._model,
                    )
                )
                self._supervisors[key] = supervisor
        await supervisor.start()
        return supervisor

    async def close(self) -> None:
        async with self._lock:
            supervisors = list(self._supervisors.values())
            self._supervisors.clear()
        await asyncio.gather(
            *(supervisor.close() for supervisor in supervisors),
            return_exceptions=True,
        )


bearer = HTTPBearer(auto_error=False)
session_registry = SessionRegistry()
router = APIRouter()


async def session_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer),
    ],
) -> str:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


async def current_supervisor(
    key: Annotated[str, Depends(session_key)],
) -> AppSupervisor:
    return await session_registry.get(key)


@router.get(
    "/session",
    response_model=AppSnapshot,
)
async def get_session(
    supervisor: Annotated[
        AppSupervisor,
        Depends(current_supervisor),
    ],
) -> AppSnapshot:
    return await supervisor.snapshot()


@router.post(
    "/session/messages",
    response_model=AppSnapshot,
    status_code=status.HTTP_202_ACCEPTED,
)
async def send_message(
    request: SendMessageRequest,
    supervisor: Annotated[
        AppSupervisor,
        Depends(current_supervisor),
    ],
) -> AppSnapshot:
    content = request.content.strip()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Message cannot be empty",
        )
    try:
        return await supervisor.send(content)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get("/session/events")
async def session_events(
    supervisor: Annotated[
        AppSupervisor,
        Depends(current_supervisor),
    ],
) -> StreamingResponse:
    subscriber = await supervisor.subscribe()

    async def events() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    snapshot = await asyncio.wait_for(subscriber.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield (
                    f"id: {snapshot.revision}\n"
                    "event: session\n"
                    f"data: {snapshot.model_dump_json()}\n\n"
                )
        finally:
            await supervisor.unsubscribe(subscriber)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
