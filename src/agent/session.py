"""One research session: who is researching, with what, over which period.

The agent layer never decides holdout access. Time lockout and testing-budget
enforcement belong to Member 4's harness; the agent only passes requests through
to the ports, which must refuse anything outside the research period or budget.
"""

import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, get_args

from agent.finding import FindingClaim
from agent.notebook import Notebook
from agent.ports import Backtester, EventQuery, Retriever
from agent.provenance import code_version
from agent.trace import Trace

# R0 (grid search) is not an agent; R1 is single-shot and does not use the loop.
AgentSystem = Literal["R1", "R2", "R3", "R3-U"]
NEEDS_RETRIEVER = {"R3", "R3-U"}


@dataclass
class ResearchSession:
    system: AgentSystem
    config: dict[str, Any]
    universe_id: str
    research_start: date
    research_end: date
    backtester: Backtester
    events: EventQuery
    notebook: Notebook
    trace: Trace
    retriever: Retriever | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    code_version: dict | None = None
    claims: list[FindingClaim] = field(default_factory=list)
    signals: dict[str, dict] = field(default_factory=dict)  # signal_id -> spec it was defined with
    tested_signals: set[str] = field(default_factory=set)  # signal_ids run_backtest has run
    done: bool = False
    invalid_calls: int = 0
    step: int = 0

    @property
    def agent_config(self) -> dict[str, Any]:
        return self.config.get("agent", {})

    def log_start(self, system_prompt: str) -> dict:
        agent = self.agent_config
        return self.trace.session_start(
            self.config,
            system_prompt,
            self.universe_id,
            agent.get("model"),
            agent.get("max_tokens"),
            agent.get("effort"),
            code_version=self.code_version,
            agent_system=self.system,
            research_start=self.research_start,
            research_end=self.research_end,
        )


def _as_date(value: date | str | None, name: str) -> date:
    if value is None:
        raise ValueError(
            f"{name} is not set. The team has not agreed the research/holdout split yet "
            f"(configs/default.yaml); pass {name} explicitly."
        )
    return value if isinstance(value, date) else date.fromisoformat(value)


def new_session(
    system: str,
    config: dict[str, Any],
    universe_id: str,
    backtester: Backtester,
    events: EventQuery,
    retriever: Retriever | None = None,
    research_start: date | str | None = None,
    research_end: date | str | None = None,
    in_memory: bool = False,
) -> ResearchSession:
    """Build a session. Explicit research dates override the config.

    TODO(member-3): drop the research_end override once the team agrees the split
    and configs/default.yaml has research_end set.
    """
    if system not in get_args(AgentSystem):
        raise ValueError(f"system must be one of {get_args(AgentSystem)}, got {system!r}")
    if system in NEEDS_RETRIEVER and retriever is None:
        raise ValueError(f"{system} uses retrieval; pass a retriever")
    start = _as_date(research_start or config.get("research_start"), "research_start")
    end = _as_date(research_end or config.get("research_end"), "research_end")
    if end <= start:
        raise ValueError(f"research_end {end} must be after research_start {start}")

    session_id = uuid.uuid4().hex
    trace_dir = None if in_memory else config.get("agent", {}).get("trace_dir")
    trace = Trace(session_id, Path(trace_dir) if trace_dir else None)
    return ResearchSession(
        system=system,
        config=config,
        universe_id=universe_id,
        research_start=start,
        research_end=end,
        backtester=backtester,
        events=events,
        retriever=retriever,
        notebook=Notebook(trace),
        trace=trace,
        session_id=session_id,
        code_version=code_version(),
    )
