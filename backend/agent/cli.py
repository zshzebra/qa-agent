import asyncio
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.google import GoogleModel

from agent.telemetry import configure_telemetry


async def main() -> None:
    load_dotenv()


    configure_telemetry()

    from agent.system import AgentSystem
    from onboarding import ExampleCompanyOnboarding

    model = GoogleModel("gemini-3.1-flash-lite-preview")
    session_file = Path("session.json")
    restored = session_file.exists()

    if restored:
        system = AgentSystem.deserialize(
            session_file.read_text(encoding="utf-8"),
            output_type=ExampleCompanyOnboarding,
            planner_model=model,
            speaker_model=model,
            researcher_model=model,
            auditor_model=model,
        )

        for message in system.dialogue:
            if message.role == "user":
                print(f"> {message.content}")
            else:
                print(message.content)
    else:
        system = AgentSystem(
            output_type=ExampleCompanyOnboarding,
            planner_model=model,
            speaker_model=model,
            researcher_model=model,
            auditor_model=model,
        )

    await system.run()

    try:
        # On restoration, the previous output queue no longer exists.
        # Use the persisted waiting state to resume directly at input().
        if (
            restored
            and system.waiting_for_user
            and not system.is_done
            and system.failure is None
        ):
            answer = await asyncio.to_thread(input, "> ")
            await system.send(answer)

        while True:
            output = await system.receive()

            if output.kind == "message":
                print(f"\n{output.message}")

                if output.expects_reply:
                    answer = await asyncio.to_thread(input, "> ")
                    await system.send(answer)

            elif output.kind == "state":
                if output.expects_reply:
                    answer = await asyncio.to_thread(input, "> ")
                    await system.send(answer)

            elif output.kind == "done":
                print(f"\n{output.message}\n\nFinal plan:\n")
                print(output.final_plan)
                return

            else:
                raise RuntimeError(output.error)

    except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
        pass

    finally:
        # Stop tasks first so interrupted research jobs return to "queued".
        await system.close()

        saved_json = await system.serialize()
        session_file.write_text(
            saved_json,
            encoding="utf-8",
        )


if __name__ == "__main__":
    asyncio.run(main())
