import argparse
import asyncio
import json
import os
from typing import Literal, TypedDict, cast

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Snapshot(TypedDict):
    revision: int
    status: Literal["working", "waiting_for_user", "completed", "failed"]
    messages: list[Message]
    completion_message: str | None
    final_plan: str | None
    error: str | None


console = Console()


def render_new_messages(snapshot: Snapshot, rendered: int) -> int:
    messages = snapshot["messages"]
    if len(messages) < rendered:
        rendered = 0

    for message in messages[rendered:]:
        if message["role"] == "user":
            console.print(f"[bold cyan]>[/bold cyan] {escape(message['content'])}")
        else:
            console.print()
            console.print(Markdown(message["content"]))

    return len(messages)


async def get_snapshot(client: httpx.AsyncClient) -> Snapshot:
    response = await client.get("session")
    response.raise_for_status()
    return cast(Snapshot, response.json())


async def send_message(client: httpx.AsyncClient, content: str) -> Snapshot:
    response = await client.post("session/messages", json={"content": content})
    response.raise_for_status()
    return cast(Snapshot, response.json())


async def wait_for_update(
    client: httpx.AsyncClient,
    revision: int,
) -> Snapshot:
    async with client.stream("GET", "session/events") as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            snapshot = cast(Snapshot, json.loads(line.removeprefix("data: ")))
            if snapshot["revision"] > revision:
                return snapshot
    raise RuntimeError("The event stream closed")


async def main(url: str, key: str) -> None:
    headers = {"Authorization": f"Bearer {key}"}
    timeout = httpx.Timeout(120, read=None)

    async with httpx.AsyncClient(
        base_url=f"{url.rstrip('/')}/",
        headers=headers,
        timeout=timeout,
    ) as client:
        snapshot = await get_snapshot(client)
        rendered = render_new_messages(snapshot, 0)

        while True:
            if snapshot["status"] == "failed":
                raise RuntimeError(snapshot["error"] or "The session failed")

            if snapshot["status"] == "completed":
                if snapshot["completion_message"]:
                    console.print()
                    console.print(Markdown(snapshot["completion_message"]))
                if snapshot["final_plan"]:
                    console.print()
                    console.rule("Final plan")
                    console.print(Markdown(snapshot["final_plan"]))
                return

            if snapshot["status"] == "waiting_for_user":
                answer = console.input("[bold cyan]>[/bold cyan] ")
                snapshot = await send_message(client, answer)
                rendered = len(snapshot["messages"])
                continue

            snapshot = await wait_for_update(client, snapshot["revision"])
            rendered = render_new_messages(snapshot, rendered)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", nargs="?", default="cli")
    parser.add_argument(
        "--url",
        default=os.getenv("LEAD_FORM_API_URL", "http://localhost:8020/api"),
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.url, args.key))
    except (EOFError, KeyboardInterrupt):
        pass
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        console.print(
            f"[bold red]Request failed ({exc.response.status_code}):[/bold red] "
            f"{escape(str(detail))}"
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")


if __name__ == "__main__":
    run()
