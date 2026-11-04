"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

The six gated tools over the interim ports. Argument models are built per universe from
its fields, text features, breakdowns and holding period, so the schemas the model sees
only offer what the data has. register_interim_tools() replaces the fake tools in the
registry (one session at a time, as the runner runs them).
"""

from datetime import date
from typing import Any, Literal

from pydantic import Field

from agent.fakes import define_signal, inspect_results, query_events
from agent.interim.backtest import TESTED, rounded
from agent.interim.spec import SpecModels
from agent.tool_descriptions import DESCRIPTIONS
from agent.tools import RAG_SYSTEMS, Tool, ToolArgs, register, unregister

EXAMPLE_EVENT_IDS = 3
TEXT_FEATURE_EXAMPLES = 5
SEARCH_K_DEFAULT, SEARCH_K_MAX = 3, 5
MODEL_BACKTEST_KEYS = ("signal_id", "status", "n_events", "n_unmeasurable", "mean_car",
                       "position_mean_car", "t_stat", "p_value_raw", "hit_rate",
                       "tests_used", "budget_remaining")

DESCRIPTIONS_INTERIM = {
    "run_backtest": (
        "Run the event study for a defined signal. USES ONE UNIT OF THE TESTING BUDGET, "
        "whatever the outcome. Returns status, n_events (events with a measurable CAR), "
        "mean_car (the raw mean abnormal return after the events, whatever the "
        "direction), position_mean_car (the signal's return: mean_car for long, minus "
        "mean_car for short), t-stat, raw p-value, hit rate of the position, tests used, "
        "budget remaining, warnings and up to 3 example event_ids."
    ),
    "inspect_results": (
        "Break down a backtested signal's results (see the task brief for the breakdowns) "
        "to check robustness. Free."
    ),
    "search_announcements": (
        DESCRIPTIONS["search_announcements"]
        + f" k is at most {SEARCH_K_MAX}. Each hit lists the event_ids of its filing."
    ),
}
MASKED_DATES_NOTE = " Dates in passages and quotes are replaced by [DATE]."


def backtest_view(result: dict[str, Any], min_events: int | None) -> dict[str, Any]:
    """What the model sees: rounded stats and budget, warnings, a few example event_ids."""
    view = rounded({k: result.get(k) for k in MODEL_BACKTEST_KEYS})
    warnings = []
    if result.get("duplicate_of"):
        warnings.append(f"duplicate of {result['duplicate_of']}: same events; not counted")
    if result["status"] != TESTED:
        warnings.append(f"not testable ({result['status']}); cannot be a finding")
    if min_events and result["n_events"] < min_events:
        warnings.append(f"n_events below the minimum of {min_events}; cannot be a finding")
    if result["n_unmeasurable"]:
        warnings.append(f"{result['n_unmeasurable']} selected events have no measurable CAR")
    view["warnings"] = warnings
    view["example_event_ids"] = result["event_ids"][:EXAMPLE_EVENT_IDS]
    return view


def run_backtest(session, args) -> dict:
    result = session.backtester.run(args.signal_id)  # full result, every event_id
    session.tested_signals.add(args.signal_id)  # only reached on success
    session.signal_event_ids[args.signal_id] = list(result["event_ids"])
    session.call_extra["event_ids"] = list(result["event_ids"])  # trace only, not the model
    session.call_extra["status"] = result["status"]
    session.call_extra["duplicate_of"] = result.get("duplicate_of")
    view = backtest_view(result, session.config.get("min_events"))
    summary = (f"{args.signal_id}: n={view['n_events']}, mean CAR={view['mean_car']}, "
               f"t={view['t_stat']}, {view['status']}")
    data = {k: v for k, v in view.items() if k != "example_event_ids"}
    session.notebook.add("backtest_result", session.step, summary, data)
    return view


def search_announcements(session, args) -> dict:
    filters = {"ticker": args.ticker, "event_type": args.event_type, "start": args.start,
               "end": args.end}
    return {"results": session.retriever.search(args.query, filters, args.k)}


def get_text_feature(session, args) -> dict:
    """Label counts over the requested events that have the feature, plus examples."""
    values = session.retriever.text_feature(args.event_ids, args.feature)
    labelled = [v for v in values if v["label"] is not None]
    counts: dict[str, int] = {}
    for v in labelled:
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    return {"feature": args.feature, "n_events": len(labelled),
            "n_without_feature": len(values) - len(labelled), "label_counts": counts,
            "examples": labelled[:TEXT_FEATURE_EXAMPLES]}


def interim_tools(
    models: SpecModels,
    event_types: tuple[str, ...],
    breakdowns: tuple[str, ...],
    text_features: tuple[str, ...],
    mask_dates: bool = False,
) -> list[Tool]:
    types = Literal[event_types]  # type: ignore[valid-type]

    class QueryEventsArgs(ToolArgs):
        event_type: types  # type: ignore[valid-type]
        conditions: list[models.condition] = Field(  # type: ignore[name-defined]
            default_factory=list, max_length=5)
        limit: int = Field(10, ge=0, le=20)

    class DefineSignalArgs(ToolArgs):
        spec: models.signal_spec  # type: ignore[name-defined]
        hypothesis_id: str | None = Field(None, description="The hypothesis this signal tests.")

    class SignalIdArgs(ToolArgs):
        signal_id: str

    class InspectResultsArgs(ToolArgs):
        signal_id: str
        by: Literal[breakdowns]  # type: ignore[valid-type]

    class SearchArgs(ToolArgs):
        query: str = Field(min_length=1)
        ticker: str | None = None
        event_type: types | None = None  # type: ignore[valid-type]
        start: date | None = Field(None, description="Earliest filing date, YYYY-MM-DD.")
        end: date | None = Field(None, description="Latest filing date, YYYY-MM-DD.")
        k: int = Field(SEARCH_K_DEFAULT, ge=1, le=SEARCH_K_MAX)

    tools = [
        Tool("query_events", QueryEventsArgs, query_events),
        Tool("define_signal", DefineSignalArgs, define_signal),
        Tool("run_backtest", SignalIdArgs, run_backtest,
             description=DESCRIPTIONS_INTERIM["run_backtest"]),
        Tool("inspect_results", InspectResultsArgs, inspect_results,
             description=DESCRIPTIONS_INTERIM["inspect_results"]),
        Tool("search_announcements", SearchArgs, search_announcements, systems=RAG_SYSTEMS,
             description=DESCRIPTIONS_INTERIM["search_announcements"]
             + (MASKED_DATES_NOTE if mask_dates else "")),
    ]
    if text_features:

        class TextFeatureArgs(ToolArgs):
            event_ids: list[str] = Field(min_length=1, max_length=200)
            feature: Literal[text_features]  # type: ignore[valid-type]

        tools.append(Tool("get_text_feature", TextFeatureArgs, get_text_feature,
                          systems=RAG_SYSTEMS))
    return tools


def register_interim_tools(*args, **kwargs) -> list[Tool]:
    """Register interim_tools(...) in place of the fake ones. A universe without text
    features gets no get_text_feature tool, so an earlier registration is removed."""
    tools = interim_tools(*args, **kwargs)
    if not any(t.name == "get_text_feature" for t in tools):
        unregister("get_text_feature")
    return [register(t) for t in tools]
