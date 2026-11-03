import copy
import json

from agent.fakes import PLANTED_SPEC
from agent.llm import FakeLLM, LLMResponse, ToolCall
from agent.loop import run_session
from agent.prompts import SYSTEM_PROMPT, build_task_brief, prompt_version
from agent.tools import dispatch, is_grounded, schemas_for

SPEC = {k: PLANTED_SPEC[k] for k in ("event_type", "conditions")}


# --- prompt_version ------------------------------------------------------------------------


def test_prompt_version_is_a_stable_short_hash(fake_session):
    s = fake_session("R2")
    brief, tools = build_task_brief(s), schemas_for("R2")
    version = prompt_version(SYSTEM_PROMPT, brief, tools)
    assert len(version) == 12 and int(version, 16) >= 0
    assert prompt_version(SYSTEM_PROMPT, brief, copy.deepcopy(tools)) == version


def test_changing_a_tool_description_changes_the_hash(fake_session):
    s = fake_session("R2")
    brief, tools = build_task_brief(s), schemas_for("R2")
    edited = copy.deepcopy(tools)
    edited[0]["description"] += " Edited."
    assert prompt_version(SYSTEM_PROMPT, brief, edited) != prompt_version(SYSTEM_PROMPT, brief,
                                                                           tools)
    assert prompt_version(SYSTEM_PROMPT + " ", brief, tools) != prompt_version(SYSTEM_PROMPT,
                                                                                brief, tools)


def test_session_start_records_prompt_version(fake_session):
    s = fake_session("R2")
    finish = ToolCall(id="t", name="finish", arguments={"reason": "done"})
    run_session(s, FakeLLM([LLMResponse(tool_calls=[finish], stop_reason="tool_use")]))
    start = s.trace.events[0]
    assert start["prompt_version"] == prompt_version(SYSTEM_PROMPT, build_task_brief(s),
                                                     schemas_for("R2"))


# --- is_grounded ---------------------------------------------------------------------------


def test_grounding_is_exact_substring_after_collapsing_whitespace():
    corpus = ["[FAKE] repurchase up to $196 million"]
    assert is_grounded("repurchase   up to\n$196", corpus)
    assert not is_grounded("repurchase up to $200", corpus)
    assert not is_grounded("   ", corpus)


# --- Through report_finding ---------------------------------------------------------------


def call(s, name, **arguments):
    content, is_error = dispatch(s, name, arguments)
    assert not is_error, content
    return json.loads(content)


def report(s, evidence):
    content, is_error = dispatch(s, "report_finding",
                                 {"signal_id": "sig-1", "rationale": "r", "evidence": evidence})
    assert not is_error, content
    assert "grounded" not in content  # measured, never shown to the model
    return s.notebook.entries[-1].data["evidence"]


def backtested(s, spec=SPEC):
    call(s, "define_signal", spec=spec)
    call(s, "run_backtest", signal_id="sig-1")
    return s.signal_event_ids["sig-1"]


def test_event_evidence_is_grounded_by_the_signal_event_set(fake_session, events):
    s = fake_session("R2")
    in_signal = backtested(s)[5]  # not one of the 3 example ids the model saw
    outside = events[(events["mcap_bucket"] == "large")
                     & (events["first_tradeable_date"] <= s.research_end)]["event_id"].iloc[0]
    evidence = report(s, [
        {"event_id": in_signal, "quote": "any wording"},
        {"event_id": outside, "quote": "any wording"},
        {"event_id": "0000000000-99-000000-dividend-1", "quote": "any wording"},
    ])
    assert [(q["grounding_method"], q["grounded"], q["event_exists"], q["in_signal"])
            for q in evidence] == [
        ("event", True, True, True),
        ("event", False, True, False),
        ("event", False, False, False),
    ]
    totals = s.trace.totals()
    assert (totals["evidence_quotes_event"], totals["evidence_quotes_event_grounded"]) == (3, 1)
    assert totals["evidence_quotes_text"] == 0


def test_holdout_events_do_not_exist_for_the_agent(fake_session, events):
    s = fake_session("R2")
    backtested(s)
    later = events[events["first_tradeable_date"] > s.research_end]["event_id"].iloc[0]
    evidence = report(s, [{"event_id": later, "quote": "q"}])
    assert evidence[0]["event_exists"] is False and evidence[0]["grounded"] is False


def test_doc_ref_evidence_is_grounded_by_text(fake_session):
    s = fake_session("R3")
    hits = call(s, "search_announcements", query="dividend share", k=3)["results"]
    call(s, "record_hypothesis", hypothesis="my own words", rationale="r")
    backtested(s)
    evidence = report(s, [
        {"doc_ref": hits[0]["doc_ref"], "quote": "  " + hits[0]["text"].replace(" ", "  ")},
        {"doc_ref": hits[0]["doc_ref"], "quote": "dividend_change_pct: 12.63"},  # paraphrase
        {"doc_ref": hits[0]["doc_ref"], "quote": "my own words"},  # only in the model's args
    ])
    assert [(q["grounding_method"], q["grounded"]) for q in evidence] == [
        ("text", True), ("text", False), ("text", False),
    ]
    event = next(e for e in s.trace.events if e.get("name") == "report_finding")
    assert event["evidence_grounded"] == [True, False, False]
    assert event["evidence_methods"] == ["text", "text", "text"]
    totals = s.trace.totals()
    assert (totals["evidence_quotes"], totals["evidence_quotes_grounded"]) == (3, 1)
    assert (totals["evidence_quotes_text"], totals["evidence_quotes_text_grounded"]) == (3, 1)


def test_json_escaped_text_is_still_grounded(fake_session, registry):
    from agent.tools import Tool, ToolArgs, register

    class NoArgs(ToolArgs):
        pass

    register(Tool("inspect_results", NoArgs, lambda s, a: {"text": 'CEO said "buy back"'}))
    s = fake_session("R2")
    call(s, "inspect_results")
    backtested(s)
    evidence = report(s, [{"doc_ref": "X/1/ex99-1.txt", "quote": 'said "buy back"'}])
    assert evidence[0]["grounded"] is True


def test_error_results_do_not_ground_quotes(fake_session):
    s = fake_session("R2")
    dispatch(s, "run_backtest", {"signal_id": "sig-9"})  # error mentions 'sig-9'
    backtested(s)
    evidence = report(s, [{"doc_ref": "X/1/ex99-1.txt", "quote": "unknown signal_id 'sig-9'"}])
    assert evidence[0]["grounded"] is False
