"""Tool registry and dispatcher.

A Tool is a name, a pydantic model for its arguments, a handler and the research
systems allowed to use it. Descriptions live in agent/tool_descriptions.py.

Implemented here (they touch nothing outside the session):
    record_hypothesis, view_notebook, report_finding, finish

Gated on other members' contracts, registered by agent/fakes.py for now:
    query_events          Member 1 (event store)
    search_announcements  Member 2 (retrieval)         R3 and R3-U only
    get_text_feature      Member 2 (text features)     R3 and R3-U only
    define_signal         Member 4 (signal DSL)
    run_backtest          Member 4 (backtest engine)   must add to session.tested_signals
    inspect_results       Member 4 (backtest engine)
"""

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from agent.finding import EvidenceQuote, FindingClaim
from agent.llm import ToolCall
from agent.session import ResearchSession
from agent.tool_descriptions import DESCRIPTIONS

AGENT_SYSTEMS = frozenset({"R2", "R3", "R3-U"})  # R1 does not use the loop
RAG_SYSTEMS = frozenset({"R3", "R3-U"})

TRUNCATED = "[truncated]"
DEFAULT_MAX_CHARS = 4000

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ToolError(Exception):
    """An intentional rejection the agent should see and can recover from
    (unknown signal, budget used up, ...). Any other exception is a bug and propagates."""


class ToolArgs(BaseModel):
    """Base for tool arguments: unknown fields are an error the model can fix."""

    model_config = ConfigDict(extra="forbid")


Handler = Callable[[ResearchSession, Any], dict | str]


@dataclass
class Tool:
    name: str
    args_model: type[BaseModel]
    handler: Handler
    systems: frozenset[str] = AGENT_SYSTEMS
    description: str = field(default="")


_TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Add (or replace) a tool. Its description comes from tool_descriptions.py."""
    if not tool.description:
        if tool.name not in DESCRIPTIONS:
            raise KeyError(f"no description for tool {tool.name!r} in tool_descriptions.py")
        tool.description = DESCRIPTIONS[tool.name]
    _TOOLS[tool.name] = tool
    return tool


def tools_for(system: str) -> list[Tool]:
    return [t for _, t in sorted(_TOOLS.items()) if system in t.systems]


# --- Schemas ------------------------------------------------------------------


def inline_defs(schema: dict) -> dict:
    """Replace every {"$ref": "#/$defs/X"} with the definition itself and drop $defs."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return walk({**copy.deepcopy(defs[name]), **siblings})
        return {k: walk(v) for k, v in node.items() if k != "$defs"}

    return walk(schema)


def schemas_for(system: str) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": inline_defs(t.args_model.model_json_schema()),
        }
        for t in tools_for(system)
    ]


# --- Dispatch -----------------------------------------------------------------


def format_validation_error(name: str, error: ValidationError) -> str:
    """One short clause per bad field; pydantic's Literal messages list the allowed values."""
    parts = []
    for e in error.errors(include_url=False):
        where = ".".join(str(p) for p in e["loc"]) or "arguments"
        parts.append(f"{where}: {e['msg']}")
    return f"Invalid arguments for {name}: " + "; ".join(parts)


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - len(TRUNCATED), 0)] + TRUNCATED


def dispatch(session: ResearchSession, name: str, arguments: Any) -> tuple[str, bool]:
    """Run one tool call. Returns (content, is_error).

    Agent-facing errors (unknown tool, invalid arguments, ToolError) are returned and
    counted in session.invalid_calls; every other exception propagates.
    """
    started = time.perf_counter()
    extra = {}
    if name == "run_backtest":  # protocol adherence, measured not enforced
        extra["hypothesis_recorded"] = any(e.kind == "hypothesis" for e in session.notebook.entries)

    tool = _TOOLS.get(name)
    if tool is None or session.system not in tool.systems:
        available = ", ".join(t.name for t in tools_for(session.system))
        content, is_error = f"Unknown tool {name!r}. Available tools: {available}", True
    else:
        try:
            args = tool.args_model.model_validate(arguments)
            result = tool.handler(session, args)
            content = result if isinstance(result, str) else json.dumps(result, default=str)
            is_error = False
        except ValidationError as e:
            content, is_error = format_validation_error(name, e), True
        except ToolError as e:
            content, is_error = f"Error in {name}: {e}", True

    content = _cap(content, session.agent_config.get("tool_result_max_chars", DEFAULT_MAX_CHARS))
    if is_error:
        session.invalid_calls += 1
    duration_ms = (time.perf_counter() - started) * 1000
    session.trace.tool_call(
        session.step, name, arguments, content, is_error, duration_ms, **extra
    )
    return content, is_error


def dispatch_calls(session: ResearchSession, calls: list[ToolCall]) -> list[dict]:
    """Dispatch one turn's tool calls strictly in order; returns tool_results entries."""
    results = []
    for call in calls:
        content, is_error = dispatch(session, call.name, call.arguments)
        results.append({"tool_call_id": call.id, "content": content, "is_error": is_error})
    return results


# --- The agent's own tools ----------------------------------------------------


class RecordHypothesisArgs(ToolArgs):
    hypothesis: NonEmpty
    rationale: NonEmpty = Field(description="The economic reason the effect should exist.")


def record_hypothesis(session: ResearchSession, args: RecordHypothesisArgs) -> dict:
    session.notebook.add("hypothesis", session.step, args.hypothesis, {"rationale": args.rationale})
    n = sum(e.kind == "hypothesis" for e in session.notebook.entries)
    return {"recorded": True, "hypotheses_so_far": n}


class ViewNotebookArgs(ToolArgs):
    last_n: int | None = Field(default=None, ge=1)


def view_notebook(session: ResearchSession, args: ViewNotebookArgs) -> str:
    return session.notebook.render(max_entries=args.last_n)


class ReportFindingArgs(ToolArgs):
    signal_id: NonEmpty
    rationale: NonEmpty
    evidence: list[EvidenceQuote]


def report_finding(session: ResearchSession, args: ReportFindingArgs) -> dict:
    sid = args.signal_id
    if sid not in session.signals:
        known = ", ".join(sorted(session.signals)) or "none yet"
        raise ToolError(f"unknown signal_id {sid!r}. Defined signals: {known}")
    if sid not in session.tested_signals:
        raise ToolError(f"signal {sid!r} was never backtested; call run_backtest first")
    if any(c.signal_id == sid for c in session.claims):
        raise ToolError(f"a finding for {sid!r} was already reported; one claim per signal")

    claim = FindingClaim(
        signal_id=sid, rationale=args.rationale, evidence=args.evidence, step=session.step
    )
    session.claims.append(claim)
    session.notebook.add(
        "claim", session.step, f"{sid}: {args.rationale}",
        {"signal_id": sid, "n_evidence": len(args.evidence)},
    )
    return {"accepted": True, "claims_so_far": len(session.claims)}


class FinishArgs(ToolArgs):
    reason: NonEmpty


def finish(session: ResearchSession, args: FinishArgs) -> dict:
    session.done = True
    session.notebook.add("note", session.step, f"finish: {args.reason}", {"reason": args.reason})
    return {"finished": True, "claims": len(session.claims)}


for _tool in (
    Tool("record_hypothesis", RecordHypothesisArgs, record_hypothesis),
    Tool("view_notebook", ViewNotebookArgs, view_notebook),
    Tool("report_finding", ReportFindingArgs, report_finding),
    Tool("finish", FinishArgs, finish),
):
    register(_tool)
