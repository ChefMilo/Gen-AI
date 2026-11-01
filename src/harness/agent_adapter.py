"""Agent tools backed by the real event-study engine and guardrail harness."""

from typing import Literal

from pydantic import Field

from agent.finding import Finding
from agent.session import ResearchSession
from agent.tools import (
    RAG_SYSTEMS,
    Tool,
    ToolArgs,
    ToolError,
    check_hypothesis_id,
    record_signal,
    register,
)
from dsl.condition import Condition
from dsl.signal import SignalSpec
from harness.backtester import GuardrailError, ResearchBacktester
from harness.data_access import ResearchEventQuery, ResearchRetriever
from store.schema import EventType


class DefineSignalArgs(ToolArgs):
    spec: SignalSpec
    hypothesis_id: str | None = None


class SignalIdArgs(ToolArgs):
    signal_id: str = Field(min_length=1)


class InspectResultsArgs(SignalIdArgs):
    by: str = Field(min_length=1)


def define_signal(session: ResearchSession, args: DefineSignalArgs) -> dict:
    check_hypothesis_id(session, args.hypothesis_id)
    if args.spec.text_filter is not None and session.system not in RAG_SYSTEMS:
        raise ToolError("Text filters require a retrieval-enabled system")
    spec = args.spec.model_dump(mode="json")
    try:
        signal_id = session.backtester.define(spec)
    except GuardrailError as error:
        raise ToolError(str(error)) from error
    record_signal(session, signal_id, spec, args.hypothesis_id)
    return {"signal_id": signal_id, "hypothesis_id": args.hypothesis_id}


def run_backtest(session: ResearchSession, args: SignalIdArgs) -> dict:
    try:
        result = session.backtester.run(args.signal_id)
    except GuardrailError as error:
        raise ToolError(str(error)) from error

    event_ids = list(result["event_ids"])
    session.tested_signals.add(args.signal_id)
    session.signal_event_ids[args.signal_id] = event_ids
    session.call_extra["event_ids"] = event_ids
    keys = (
        "signal_id",
        "status",
        "n_selected",
        "n_events",
        "mean_car",
        "directional_mean_car",
        "t_stat",
        "p_value_raw",
        "hit_rate",
        "tests_used",
        "budget_remaining",
        "budget_enforced",
    )
    view = {key: result.get(key) for key in keys}
    view["example_event_ids"] = event_ids[:3]
    if result.get("portfolio") is not None:
        view["portfolio"] = result["portfolio"]["summary"]
    session.notebook.add(
        "backtest_result",
        session.step,
        f"{args.signal_id}: n={result['n_events']}, mean CAR={result['mean_car']}",
        {key: value for key, value in view.items() if key != "example_event_ids"},
    )
    return view


def inspect_results(session: ResearchSession, args: InspectResultsArgs) -> dict:
    try:
        return session.backtester.breakdown(args.signal_id, args.by)
    except GuardrailError as error:
        raise ToolError(str(error)) from error


def register_backtest_tools() -> None:
    """Replace only the three backtest registrations; preserve other ports.

    Call after temporary-tool registration, which otherwise overwrites these.
    This uses the agent's process-global registry; sessions run sequentially.
    """
    for tool in (
        Tool("define_signal", DefineSignalArgs, define_signal),
        Tool("run_backtest", SignalIdArgs, run_backtest),
        Tool("inspect_results", InspectResultsArgs, inspect_results),
    ):
        register(tool)


def finalize_agent_session(session: ResearchSession) -> list[Finding]:
    """Controller-only finalization after normal completion or a resource cap."""
    if not isinstance(session.backtester, ResearchBacktester):
        raise TypeError("Finalization requires ResearchBacktester")
    if not session.trace.events or session.trace.events[-1]["kind"] != "session_end":
        raise GuardrailError("Finish the agent loop before finalizing findings")
    if session.trace.events[-1].get("reason") == "error":
        raise GuardrailError("Cannot finalize an agent session that ended with an error")

    if isinstance(session.events, ResearchEventQuery):
        config = session.backtester.configuration
        if (
            session.events.configuration != config
            or session.research_start != config.research_start
            or session.research_end != config.research_end
        ):
            raise GuardrailError("Session data ports disagree with the harness research period")
        try:
            for claim in session.claims:
                for evidence in claim.evidence:
                    session.events.validate_event_evidence(
                        evidence, session.signal_event_ids.get(claim.signal_id, [])
                    )
                    if evidence.doc_ref is not None:
                        if not isinstance(session.retriever, ResearchRetriever):
                            raise GuardrailError("Document evidence requires the bounded retriever")
                        session.retriever.validate_document_evidence(evidence)
        except GuardrailError as error:
            session.backtester._record("claim_evidence_rejected", error=str(error))
            raise

    return session.backtester.finalize(
        claims=[claim.model_dump(mode="json") for claim in session.claims],
        session_id=session.session_id,
        system=session.system,
    )


class QueryEventsArgs(ToolArgs):
    event_type: EventType
    conditions: list[Condition] = Field(default_factory=list, max_length=5)
    limit: int = Field(default=10, strict=True, ge=0, le=20)


class SearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    ticker: str | None = None
    event_type: EventType | None = None
    k: int = Field(default=5, strict=True, ge=1, le=10)


class TextFeatureArgs(ToolArgs):
    event_ids: list[str] = Field(min_length=1, max_length=200)
    feature: Literal["dividend_framing", "buyback_rationale"]


def query_events(session: ResearchSession, args: QueryEventsArgs) -> dict:
    if not isinstance(session.events, ResearchEventQuery):
        raise ToolError("Real query tools require a research-period event port")
    try:
        return session.events.query(
            args.event_type, [c.model_dump(mode="json") for c in args.conditions], args.limit
        )
    except GuardrailError as error:
        raise ToolError(str(error)) from error


def search_announcements(session: ResearchSession, args: SearchArgs) -> dict:
    if not isinstance(session.retriever, ResearchRetriever):
        raise ToolError("Real search tools require a cutoff-bound retrieval port")
    filters = {
        key: value
        for key, value in {
            "tickers": [args.ticker] if args.ticker is not None else None,
            "event_type": args.event_type,
        }.items()
        if value is not None
    }
    try:
        return {"results": session.retriever.search(args.query, filters, args.k)}
    except GuardrailError as error:
        raise ToolError(str(error)) from error


def get_text_feature(session: ResearchSession, args: TextFeatureArgs) -> dict:
    if not isinstance(session.retriever, ResearchRetriever):
        raise ToolError("Text feature tools require a cutoff-bound retrieval port")
    try:
        values = session.retriever.text_feature(args.event_ids, args.feature)
    except GuardrailError as error:
        raise ToolError(str(error)) from error
    available = [value for value in values if value["label"] is not None]
    counts = {}
    for value in available:
        counts[value["label"]] = counts.get(value["label"], 0) + 1
    return {
        "feature": args.feature,
        "n_events": len(available),
        "label_counts": counts,
        "examples": available[:5],
    }


def register_research_tools() -> None:
    """Register all six real-port tools after any temporary registrations."""
    register_backtest_tools()
    for tool in (
        Tool("query_events", QueryEventsArgs, query_events),
        Tool("search_announcements", SearchArgs, search_announcements, systems=RAG_SYSTEMS),
        Tool("get_text_feature", TextFeatureArgs, get_text_feature, systems=RAG_SYSTEMS),
    ):
        register(tool)
