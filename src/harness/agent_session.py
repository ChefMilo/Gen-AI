"""Bind agent ports to one research configuration, without Git provenance calls."""

import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent.notebook import Notebook
from agent.session import ResearchSession
from agent.trace import Trace
from harness.backtester import GuardrailError, ResearchBacktester
from harness.data_access import ResearchEventQuery, ResearchRetriever


class AgentLimits(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_steps: int = Field(default=60, strict=True, ge=1, le=1000)
    max_session_input_tokens: int = Field(default=100000, strict=True, ge=1)


def make_research_session(
    backtester: ResearchBacktester,
    *,
    system: str,
    universe_id: str,
    agent_config: dict | None = None,
    retrieval_backend=None,
    trace_dir: Path | None = None,
) -> ResearchSession:
    """Create real event/backtest ports from the exact same research snapshot.

    No subprocesses, model calls or Git commands are executed. The caller owns
    the already-created backtester and chooses the retrieval backend. A missing
    trace_dir keeps the agent trace in memory; the harness audit remains on disk.
    """
    if system not in {"R2", "R3", "R3-U"}:
        raise ValueError("This iterative-session factory supports R2, R3 and R3-U")
    config = backtester.configuration
    if (system == "R3-U") != (config.mode == "unguarded"):
        raise GuardrailError("System label and guarded/unguarded configuration disagree")
    if backtester.budget_status()["tests_used"]:
        raise GuardrailError("Bind the agent before running any backtests")
    if config.research_start >= config.research_end:
        raise ValueError("Agent research_end must be after research_start")
    rag = system in {"R3", "R3-U"}
    if rag != (retrieval_backend is not None):
        raise ValueError("Provide retrieval_backend for R3/R3-U only")

    settings = AgentLimits.model_validate(agent_config or {}).model_dump()
    events = ResearchEventQuery(backtester.research_events(), config)
    retriever = ResearchRetriever(retrieval_backend, events) if rag else None
    session_id = uuid.uuid4().hex
    trace = Trace(session_id, trace_dir)
    return ResearchSession(
        system=system,
        config={
            "research_start": config.research_start,
            "research_end": config.research_end,
            "min_events": config.min_events,
            "backtest_budget": config.budget,
            "alpha": config.alpha,
            "agent": settings,
        },
        universe_id=universe_id,
        research_start=config.research_start,
        research_end=config.research_end,
        backtester=backtester,
        events=events,
        notebook=Notebook(trace),
        trace=trace,
        retriever=retriever,
        session_id=session_id,
        code_version=None,
    )
