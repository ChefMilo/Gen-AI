import json
from typing import Literal

import pytest
from pydantic import BaseModel

import agent.tools as tools
from agent.llm import ToolCall
from agent.tool_descriptions import DESCRIPTIONS
from agent.tools import (
    TRUNCATED,
    Tool,
    ToolArgs,
    ToolError,
    dispatch,
    dispatch_calls,
    inline_defs,
    register,
    schemas_for,
    tools_for,
)

CORE = ["finish", "record_hypothesis", "report_finding", "view_notebook"]
DESIGN_DOC_TOOLS = [
    "search_announcements", "query_events", "get_text_feature", "define_signal",
    "run_backtest", "inspect_results", "report_finding",
]


class SignalArgs(ToolArgs):
    signal_id: str


class DirectionArgs(ToolArgs):
    direction: Literal["long", "short"]
    holding_days: int


def fake_define(session, args):
    session.signals[args.signal_id] = {}
    return {"signal_id": args.signal_id}


def fake_backtest(session, args):
    if args.signal_id not in session.signals:
        raise ToolError(f"unknown signal_id {args.signal_id!r}")
    session.tested_signals.add(args.signal_id)
    return {"signal_id": args.signal_id, "t_stat": 2.5}


@pytest.fixture
def fake_signal_tools(registry):
    register(Tool("define_signal", SignalArgs, fake_define))
    register(Tool("run_backtest", SignalArgs, fake_backtest))


def call(session, name, **arguments):
    return dispatch(session, name, arguments)


def error_events(session):
    return [e for e in session.trace.events if e["kind"] == "tool_call" and e["is_error"]]


# --- Registry and schemas -----------------------------------------------------


def test_core_tools_registered_for_every_agent_system():
    for system in ["R2", "R3", "R3-U"]:
        assert [t.name for t in tools_for(system)] == CORE
    assert tools_for("R1") == []


def test_descriptions_cover_design_doc_and_new_tools():
    assert set(DESCRIPTIONS) == set(DESIGN_DOC_TOOLS) | {"record_hypothesis", "view_notebook",
                                                         "finish"}
    assert all(t.description == DESCRIPTIONS[t.name] for t in tools_for("R3"))


def test_register_needs_a_description(registry):
    with pytest.raises(KeyError, match="no description"):
        register(Tool("made_up_tool", SignalArgs, fake_define))


def test_schemas_are_flat_and_api_shaped():
    schemas = {s["name"]: s for s in schemas_for("R2")}
    assert list(schemas) == CORE
    for s in schemas.values():
        assert set(s) == {"name", "description", "input_schema"}
        assert s["input_schema"]["type"] == "object"
        text = json.dumps(s["input_schema"])
        assert "$defs" not in text and "$ref" not in text
    evidence = schemas["report_finding"]["input_schema"]["properties"]["evidence"]
    assert set(evidence["items"]["properties"]) == {"event_id", "doc_ref", "quote"}
    assert schemas["finish"]["input_schema"]["additionalProperties"] is False


def test_inline_defs_handles_nested_refs():
    class Inner(BaseModel):
        x: int

    class Middle(BaseModel):
        inner: Inner
        many: list[Inner]

    class Outer(BaseModel):
        middle: Middle | None = None

    raw = Outer.model_json_schema()
    assert "$defs" in raw
    flat = inline_defs(raw)
    assert "$ref" not in json.dumps(flat) and "$defs" not in flat
    middle = flat["properties"]["middle"]["anyOf"][0]
    assert middle["properties"]["inner"]["properties"]["x"]["type"] == "integer"
    assert middle["properties"]["many"]["items"]["properties"]["x"]["type"] == "integer"


def test_rag_tools_are_hidden_from_r2(registry):
    register(Tool("search_announcements", SignalArgs, fake_define, systems=tools.RAG_SYSTEMS))
    assert "search_announcements" not in [s["name"] for s in schemas_for("R2")]
    assert "search_announcements" in [s["name"] for s in schemas_for("R3")]


# --- Dispatch errors ----------------------------------------------------------


def test_disallowed_tool_is_an_error_listing_available_tools(registry, make_session):
    register(Tool("search_announcements", SignalArgs, fake_define, systems=tools.RAG_SYSTEMS))
    s = make_session("R2")
    content, is_error = call(s, "search_announcements", signal_id="x")
    assert is_error
    assert content == (
        "Unknown tool 'search_announcements'. "
        "Available tools: finish, record_hypothesis, report_finding, view_notebook"
    )
    assert s.invalid_calls == 1


def test_unknown_tool_is_an_error(make_session):
    s = make_session()
    content, is_error = call(s, "buy_stock")
    assert is_error and content.startswith("Unknown tool 'buy_stock'. Available tools: finish")


def test_validation_errors_name_fields_and_allowed_values(registry, make_session):
    register(Tool("define_signal", DirectionArgs, lambda s, a: {}))
    s = make_session()
    content, is_error = call(s, "define_signal", direction="up", colour="red")
    assert is_error
    assert content.startswith("Invalid arguments for define_signal: ")
    assert "direction: Input should be 'long' or 'short'" in content
    assert "holding_days: Field required" in content
    assert "colour: Extra inputs are not permitted" in content
    assert len(content) < 300
    assert s.invalid_calls == 1


def test_non_object_arguments_are_a_validation_error(make_session):
    content, is_error = dispatch(make_session(), "finish", "please stop")
    assert is_error and content.startswith("Invalid arguments for finish")


def test_tool_errors_are_returned_to_the_agent(registry, make_session):
    def handler(session, args):
        raise ToolError("budget exhausted: 0 of 15 backtests left")

    register(Tool("inspect_results", SignalArgs, handler))
    s = make_session()
    content, is_error = call(s, "inspect_results", signal_id="s")
    assert is_error
    assert content == "Error in inspect_results: budget exhausted: 0 of 15 backtests left"
    assert s.invalid_calls == 1


def test_plain_key_error_propagates_and_is_not_counted(registry, make_session):
    def handler(session, args):
        return {}["missing"]  # a bug in the handler, not the agent's fault

    register(Tool("inspect_results", SignalArgs, handler))
    s = make_session()
    with pytest.raises(KeyError):
        call(s, "inspect_results", signal_id="s")
    assert s.invalid_calls == 0


@pytest.mark.parametrize("exc", [ValueError("v"), RuntimeError("r"), TypeError("t")])
def test_other_internal_errors_propagate(registry, make_session, exc):
    def handler(session, args):
        raise exc

    register(Tool("inspect_results", SignalArgs, handler))
    s = make_session()
    with pytest.raises(type(exc)):
        call(s, "inspect_results", signal_id="s")
    assert s.invalid_calls == 0


def test_results_are_capped_with_marker(registry, make_session):
    register(Tool("inspect_results", SignalArgs, lambda s, a: {"rows": ["x" * 50] * 100}))
    s = make_session(tool_result_max_chars=200)
    content, is_error = call(s, "inspect_results", signal_id="s")
    assert not is_error
    assert len(content) == 200 and content.endswith(TRUNCATED)


def test_results_are_json(registry, make_session):
    from datetime import date

    register(Tool("inspect_results", SignalArgs, lambda s, a: {"d": date(2020, 1, 2), "n": 3}))
    content, _ = call(make_session(), "inspect_results", signal_id="s")
    assert json.loads(content) == {"d": "2020-01-02", "n": 3}


# --- Trace logging ------------------------------------------------------------


def test_every_call_is_traced_including_rejected_ones(make_session):
    s = make_session()
    s.step = 3
    call(s, "record_hypothesis", hypothesis="h", rationale="r")
    call(s, "buy_stock", ticker="X")
    call(s, "finish", reason="")
    events = [e for e in s.trace.events if e["kind"] == "tool_call"]
    assert [(e["name"], e["is_error"]) for e in events] == [
        ("record_hypothesis", False), ("buy_stock", True), ("finish", True),
    ]
    assert events[1]["arguments"] == {"ticker": "X"}
    assert events[1]["result"].startswith("Unknown tool")
    assert all(e["step"] == 3 and e["duration_ms"] >= 0 for e in events)
    assert s.trace.totals()["invalid_calls"] == s.invalid_calls == 2


def test_run_backtest_records_whether_a_hypothesis_came_first(fake_signal_tools, make_session):
    s = make_session()
    call(s, "define_signal", signal_id="a")
    call(s, "define_signal", signal_id="b")
    call(s, "run_backtest", signal_id="a")
    call(s, "record_hypothesis", hypothesis="h", rationale="r")
    call(s, "run_backtest", signal_id="b")
    call(s, "run_backtest", signal_id="zzz")  # fails; not counted as a backtest
    backtests = [e for e in s.trace.events if e.get("name") == "run_backtest"]
    assert [e["hypothesis_recorded"] for e in backtests] == [False, True, True]
    other = [e for e in s.trace.events if e.get("name") in {"define_signal", "record_hypothesis"}]
    assert other and all("hypothesis_recorded" not in e for e in other)
    totals = s.trace.totals()
    assert (totals["backtests_with_hypothesis"], totals["backtests_without_hypothesis"]) == (1, 1)


def test_calls_in_one_turn_run_in_order(fake_signal_tools, make_session):
    s = make_session()
    turn = [
        ToolCall(id="t1", name="define_signal", arguments={"signal_id": "a"}),
        ToolCall(id="t2", name="run_backtest", arguments={"signal_id": "a"}),  # needs t1 first
        ToolCall(id="t3", name="finish", arguments={"reason": "done"}),
    ]
    results = dispatch_calls(s, turn)
    assert [r["tool_call_id"] for r in results] == ["t1", "t2", "t3"]
    assert [r["is_error"] for r in results] == [False, False, False]
    names = [e["name"] for e in s.trace.events if e["kind"] == "tool_call"]
    assert names == ["define_signal", "run_backtest", "finish"]


# --- The agent's own tools ----------------------------------------------------


def test_record_hypothesis_writes_the_notebook(make_session):
    s = make_session()
    s.step = 2
    content, is_error = call(s, "record_hypothesis", hypothesis="Hikes work", rationale="signal")
    assert not is_error and json.loads(content) == {"recorded": True, "hypotheses_so_far": 1}
    entry = s.notebook.entries[0]
    assert (entry.kind, entry.step, entry.summary, entry.data) == (
        "hypothesis", 2, "Hikes work", {"rationale": "signal"}
    )


def test_record_hypothesis_needs_text(make_session):
    s = make_session()
    _, is_error = call(s, "record_hypothesis", hypothesis="  ", rationale="r")
    assert is_error and s.notebook.entries == []


def test_view_notebook_returns_rendered_text(make_session):
    s = make_session()
    call(s, "record_hypothesis", hypothesis="first", rationale="r")
    call(s, "record_hypothesis", hypothesis="second", rationale="r")
    content, is_error = call(s, "view_notebook", last_n=1)
    assert not is_error
    assert content == s.notebook.render(max_entries=1)
    assert "second" in content and "first" not in content


def test_finish_needs_a_reason_and_records_it(make_session):
    s = make_session()
    for reason in ["", "   "]:
        content, is_error = call(s, "finish", reason=reason)
        assert is_error and "reason" in content
    assert not s.done and s.invalid_calls == 2

    content, is_error = call(s, "finish", reason="budget used up")
    assert not is_error and s.done
    assert s.notebook.entries[-1].summary == "finish: budget used up"
    assert s.notebook.entries[-1].data == {"reason": "budget used up"}
    notebook_events = [e for e in s.trace.events if e["kind"] == "notebook"]
    assert notebook_events[-1]["data"] == {"reason": "budget used up"}


# --- report_finding -----------------------------------------------------------

EVIDENCE = [{"event_id": "e1", "quote": "raised the dividend"}]


def report(s, signal_id):
    return call(s, "report_finding", signal_id=signal_id, rationale="why", evidence=EVIDENCE)


def test_report_finding_rejects_unknown_signal(fake_signal_tools, make_session):
    s = make_session()
    call(s, "define_signal", signal_id="a")
    content, is_error = report(s, "zzz")
    assert is_error
    assert content == "Error in report_finding: unknown signal_id 'zzz'. Defined signals: a"
    assert s.claims == [] and s.invalid_calls == 1


def test_report_finding_rejects_untested_signal(fake_signal_tools, make_session):
    s = make_session()
    call(s, "define_signal", signal_id="a")
    content, is_error = report(s, "a")
    assert is_error and "never backtested; call run_backtest first" in content
    assert s.claims == [] and s.invalid_calls == 1


def test_report_finding_accepts_once_per_signal(fake_signal_tools, make_session):
    s = make_session()
    call(s, "define_signal", signal_id="a")
    call(s, "run_backtest", signal_id="a")
    s.step = 5
    content, is_error = report(s, "a")
    assert not is_error and json.loads(content) == {"accepted": True, "claims_so_far": 1}
    claim = s.claims[0]
    assert (claim.signal_id, claim.rationale, claim.step) == ("a", "why", 5)
    assert claim.evidence[0].event_id == "e1"
    assert s.notebook.entries[-1].kind == "claim"

    content, is_error = report(s, "a")
    assert is_error and "already reported" in content
    assert len(s.claims) == 1 and s.invalid_calls == 1


def test_report_finding_checks_evidence_sources(fake_signal_tools, make_session):
    s = make_session()
    call(s, "define_signal", signal_id="a")
    call(s, "run_backtest", signal_id="a")
    content, is_error = call(
        s, "report_finding", signal_id="a", rationale="why", evidence=[{"quote": "orphan"}]
    )
    assert is_error and "event_id or a doc_ref" in content
    assert s.claims == []
