Perform a careful reconciliation of the current conversation and plan.

All work scheduled before this planner turn has finished: no research job is
queued or running, no research result is waiting to be processed, and no other
background work will continue unless you start it in this turn.

First establish exactly where the task stands:

1. Review the latest user input and verify that every meaningful answer,
   correction, uncertainty, refusal, acknowledgement, or question has been
   incorporated into the plan.
2. Review every unresolved field and decide whether information already known
   supports useful, non-duplicate research.
3. Identify any findings that need confirmation and any specific fact,
   decision, or disambiguation that only the user can provide.
4. Read the latest assistant message and decide whether it already gives the
   user the correct next action. Do not repeat a sufficient handoff.

Then leave the task able to advance:

- Dispatch useful research now when research can advance it.
- Otherwise, write a purposeful `Next message to user` directive for the exact
  confirmation or information needed.
- If the latest assistant message already provides that handoff, write a
  `Stay silent` directive that names the existing request so the conversational
  agent does not repeat it.
- Set `done=true` only when the normal completion requirements are met.

Merely editing the plan is not sufficient if it leaves no research running,
does not return the conversational floor to the user, and does not complete the
task. Do not write conversational wording, a greeting, a progress
announcement, or any mention of this reconciliation.
