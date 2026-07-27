from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI

from agent.telemetry import configure_telemetry

load_dotenv()
configure_telemetry()

from agent.api import router, session_registry


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await session_registry.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router)
