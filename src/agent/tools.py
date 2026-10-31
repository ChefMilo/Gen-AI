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
from agent.textmask import maybe_mask
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


def unregister(name: str) -> None:
    _TOOLS.pop(name, None)


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
    if name == "run_backtest":  # protocol adherence: measured, never enforced
        signal_id = arguments.get("signal_id") if isinstance(arguments, dict) else None
        extra["hypothesis_id"] = session.signal_hypotheses.get(signal_id)

    tool = _TOOLS.get(name)
    if tool is None or session.system not in tool.systems:
        available = ", ".join(t.name for t in tools_for(session.system))
        content, is_error = f"Unknown tool {name!r}. Available tools: {available}", True
    else:
        try:
            args = tool.args_model.model_validate(arguments)
            session.call_extra = {}
            result = tool.handler(session, args)
            extra |= session.call_extra  # measurements a handler wants on this trace event
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


# --- Helpers for signal tools (used by fakes.py now, real handlers later) -----


def check_hypothesis_id(session: ResearchSession, hypothesis_id: str | None) -> None:
    if hypothesis_id is not None and hypothesis_id not in session.hypothesis_ids():
        known = ", ".join(session.hypothesis_ids()) or "none yet; call record_hypothesis"
        raise ToolError(f"unknown hypothesis_id {hypothesis_id!r}. Recorded: {known}")


def record_signal(
    session: ResearchSession, signal_id: str, spec: dict, hypothesis_id: str | None
) -> None:
    """Remember a defined signal and which hypothesis it tests (or None)."""
    session.signals[signal_id] = spec
    session.signal_hypotheses[signal_id] = hypothesis_id
    linked = f" (tests {hypothesis_id})" if hypothesis_id else ""
    session.notebook.add(
        "signal_defined", session.step, f"{signal_id}{linked}",
        {"signal_id": signal_id, "hypothesis_id": hypothesis_id, "spec": spec},
    )


# --- Evidence grounding (measured, never enforced) -----------------------------


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def evidence_corpus(session: ResearchSession) -> list[str]:
    """Every successful tool result the model has seen so far: the raw text, plus each
    string inside it when it parses as JSON (so quotes are not hidden by JSON escaping)."""
    corpus = []
    for e in session.trace.events:
        if e["kind"] != "tool_call" or e["is_error"]:
            continue
        corpus.append(_collapse(e["result"]))
        try:
            corpus += [_collapse(s) for s in _strings(json.loads(e["result"]))]
        except json.JSONDecodeError:  # e.g. a truncated result
            pass
    return corpus


def is_grounded(quote: str, corpus: list[str]) -> bool:
    """Exact substring of an earlier tool result, after collapsing whitespace."""
    q = _collapse(quote)
    return bool(q) and any(q in text for text in corpus)


def ground_evidence(
    session: ResearchSession, signal_id: str, evidence: list[EvidenceQuote]
) -> list[dict]:
    """Per quote: with an event_id, the event must exist and be in the claimed signal's full
    event set ("event"); with only a doc_ref, the quote must appear in an earlier tool
    result ("text"), after the same date masking as the tool results (mask_dates_in_text)."""
    ids = [q.event_id for q in evidence if q.event_id]
    existing = session.events.existing_event_ids(ids) if ids else set()
    in_signal_ids = set(session.signal_event_ids.get(signal_id, []))
    corpus = evidence_corpus(session)
    out = []
    for q in evidence:
        if q.event_id:
            exists, in_signal = q.event_id in existing, q.event_id in in_signal_ids
            out.append({"grounding_method": "event", "grounded": exists and in_signal,
                        "event_exists": exists, "in_signal": in_signal})
        else:
            # The model only saw masked text, so a quote is masked the same way first.
            quote = maybe_mask(q.quote, session.agent_config.get("mask_dates_in_text", False))
            out.append({"grounding_method": "text", "grounded": is_grounded(quote, corpus)})
    return out


# --- The agent's own tools ----------------------------------------------------


class RecordHypothesisArgs(ToolArgs):
    hypothesis: NonEmpty
    rationale: NonEmpty = Field(description="The economic reason the effect should exist.")


def record_hypothesis(session: ResearchSession, args: RecordHypothesisArgs) -> dict:
    hypothesis_id = f"h{len(session.hypothesis_ids()) + 1}"
    session.notebook.add(
        "hypothesis", session.step, args.hypothesis,
        {"hypothesis_id": hypothesis_id, "rationale": args.rationale},
    )
    return {"recorded": True, "hypothesis_id": hypothesis_id}


class ViewNotebookArgs(ToolArgs):
    last_n: int | None = Field(default=None, ge=1)


def view_notebook(session: ResearchSession, args: ViewNotebookArgs) -> str:
    """The most recent entries that fit in one tool result (older ones are noted, not cut)."""
    limit = session.agent_config.get("tool_result_max_chars", DEFAULT_MAX_CHARS)
    n = args.last_n or len(session.notebook.entries)
    text = session.notebook.render(max_entries=n)
    while len(text) > limit and n > 1:
        n -= 1
        text = session.notebook.render(max_entries=n)
    return text


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

    grounding = ground_evidence(session, sid, args.evidence)
    claim = FindingClaim(
        signal_id=sid, rationale=args.rationale, evidence=args.evidence, step=session.step
    )
    session.claims.append(claim)
    session.notebook.add(
        "claim", session.step, f"{sid}: {args.rationale}",
        {
            "signal_id": sid,
            "hypothesis_id": session.signal_hypotheses.get(sid),
            "n_evidence": len(args.evidence),
            "evidence": [
                {**q.model_dump(), **g} for q, g in zip(args.evidence, grounding, strict=True)
            ],
        },
    )
    # measured, never shown to the model
    session.call_extra["evidence_grounded"] = [g["grounded"] for g in grounding]
    session.call_extra["evidence_methods"] = [g["grounding_method"] for g in grounding]
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
