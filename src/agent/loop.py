"""The agent loop: model turn -> tool calls -> results, until finish or a cap."""

from agent.finding import FindingClaim
from agent.llm import LLMClient, assistant_message
from agent.prompts import SYSTEM_PROMPT, build_task_brief, prompt_version
from agent.session import ResearchSession
from agent.tools import dispatch_calls, schemas_for
from harness.limits import ResourceLimitReached

NUDGE = (
    "You have not called finish. Continue your research with the tools, "
    "or call finish with a reason."
)


def run_session(session: ResearchSession, client: LLMClient, **start_extra) -> list[FindingClaim]:
    """Run one research session. End reasons: finished, no_tool_calls, refusal, step_cap,
    token_cap, error (the exception is re-raised after the trace is closed).
    start_extra goes into the session_start event (e.g. the runner's seed)."""
    agent = session.agent_config
    max_steps = agent.get("max_steps", 60)
    token_cap = agent.get("max_session_input_tokens")
    tools = schemas_for(session.system)
    brief = build_task_brief(session)
    messages = [{"role": "user", "text": brief}]
    session.log_start(SYSTEM_PROMPT, prompt_version(SYSTEM_PROMPT, brief, tools),
                      **start_extra)

    reason, nudged, input_tokens = "step_cap", False, 0
    try:
        while session.step < max_steps:
            session.step += 1
            try:
                response = client.complete(SYSTEM_PROMPT, messages, tools)
            except ResourceLimitReached as limit:
                reason = limit.reason
                break
            session.trace.model_turn(session.step, response)
            messages.append(assistant_message(response))
            input_tokens += response.usage.total_input_tokens

            if response.stop_reason == "refusal":
                reason = "refusal"
                break
            if response.tool_calls:
                results = dispatch_calls(session, response.tool_calls)
                messages.append({"role": "tool_results", "results": results})
                if session.done:
                    reason = "finished"
                    break
            elif nudged:  # second turn in a row without tools
                reason = "no_tool_calls"
                break
            else:  # one nudge per session
                nudged = True
                session.trace.nudge(session.step, NUDGE)
                messages.append({"role": "user", "text": NUDGE})

            if token_cap and input_tokens >= token_cap:
                reason = "token_cap"
                break
    except Exception:
        session.trace.session_end("error")
        raise

    session.trace.session_end(reason)
    return session.claims
