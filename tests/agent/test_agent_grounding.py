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


def test_report_finding_records_grounding_without_telling_the_model(fake_session):
    s = fake_session("R3")
    hits = call(s, "search_announcements", query="dividend share", k=3)["results"]
    real_quote = hits[0]["text"]
    call(s, "record_hypothesis", hypothesis="my own words", rationale="r")
    call(s, "define_signal", spec=SPEC)
    call(s, "run_backtest", signal_id="sig-1")
    content, is_error = dispatch(s, "report_finding", {
        "signal_id": "sig-1", "rationale": "r",
        "evidence": [
            {"doc_ref": hits[0]["doc_ref"], "quote": "  " + real_quote.replace(" ", "  ")},
            {"event_id": "e1", "quote": "dividend_change_pct: 12.63"},  # paraphrase
            {"event_id": "e2", "quote": "my own words"},  # only in the model's own arguments
        ],
    })
    assert not is_error and "grounded" not in content  # measured, not shown to the model

    claim = s.notebook.entries[-1]
    assert [q["grounded"] for q in claim.data["evidence"]] == [True, False, False]
    event = next(e for e in s.trace.events if e.get("name") == "report_finding")
    assert event["evidence_grounded"] == [True, False, False]
    totals = s.trace.totals()
    assert (totals["evidence_quotes"], totals["evidence_quotes_grounded"]) == (3, 1)


def test_json_escaped_text_is_still_grounded(fake_session, registry):
    from agent.tools import Tool, ToolArgs, register

    class NoArgs(ToolArgs):
        pass

    register(Tool("inspect_results", NoArgs, lambda s, a: {"text": 'CEO said "buy back"'}))
    s = fake_session("R2")
    call(s, "inspect_results")
    call(s, "define_signal", spec=SPEC)
    s.tested_signals.add("sig-1")
    call(s, "report_finding", signal_id="sig-1", rationale="r",
         evidence=[{"event_id": "e1", "quote": 'said "buy back"'}])
    assert s.notebook.entries[-1].data["evidence"][0]["grounded"] is True


def test_error_results_do_not_ground_quotes(fake_session):
    s = fake_session("R2")
    dispatch(s, "run_backtest", {"signal_id": "sig-9"})  # error mentions 'sig-9'
    call(s, "define_signal", spec=SPEC)
    call(s, "run_backtest", signal_id="sig-1")
    call(s, "report_finding", signal_id="sig-1", rationale="r",
         evidence=[{"event_id": "e1", "quote": "unknown signal_id 'sig-9'"}])
    assert s.notebook.entries[-1].data["evidence"][0]["grounded"] is False
