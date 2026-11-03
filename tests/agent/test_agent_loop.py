import pytest

from agent.fakes import PLANTED_SPEC
from agent.llm import FakeLLM, LLMResponse, ToolCall, Usage
from agent.loop import NUDGE, run_session
from agent.prompts import SYSTEM_PROMPT

SPEC = {k: PLANTED_SPEC[k] for k in ("event_type", "conditions")}


def turn(*calls, text="", stop_reason=None, input_tokens=100):
    tool_calls = [ToolCall(id=f"t{i}", name=name, arguments=args)
                  for i, (name, args) in enumerate(calls)]
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason or ("tool_use" if tool_calls else "end_turn"),
        usage=Usage(input_tokens=input_tokens, output_tokens=10),
    )


HYPOTHESIS = ("record_hypothesis", {"hypothesis": "Small-cap dividend hikes",
                                    "rationale": "signalling where information is scarce"})
FINISH = ("finish", {"reason": "found one robust signal"})

PLANTED_SCRIPT = [
    turn(HYPOTHESIS),
    turn(("define_signal", {"spec": SPEC, "hypothesis_id": "h1"})),
    turn(("run_backtest", {"signal_id": "sig-1"})),
    turn(("report_finding", {"signal_id": "sig-1", "rationale": "robust small-cap effect",
                             "evidence": [{"event_id": "e1", "quote": "raised"}]})),
    turn(FINISH),
]


def end_event(session):
    assert session.trace.events[-1]["kind"] == "session_end"
    return session.trace.events[-1]


def test_scripted_session_produces_claims_trace_and_notebook(fake_session):
    s = fake_session("R2")
    llm = FakeLLM(PLANTED_SCRIPT)
    claims = run_session(s, llm)

    assert [c.signal_id for c in claims] == ["sig-1"] and claims[0].step == 4
    end = end_event(s)
    assert end["reason"] == "finished"
    assert end["totals"]["steps"] == 5 and end["totals"]["tool_calls"] == 5
    assert end["totals"]["invalid_calls"] == 0
    assert end["totals"]["backtests_linked_to_hypothesis"] == 1
    assert end["totals"]["hypotheses_with_claim"] == 1
    kinds = [e["kind"] for e in s.trace.events]
    assert kinds[0] == "session_start" and kinds.count("model_turn") == 5
    assert [e.kind for e in s.notebook.entries] == [
        "hypothesis", "signal_defined", "backtest_result", "claim", "note",
    ]

    # What the model saw: system prompt, brief first, then alternating turns and results
    first, last = llm.calls[0], llm.calls[-1]
    assert first["system"] == SYSTEM_PROMPT
    assert first["messages"][0]["role"] == "user"
    assert "Backtest budget: 15" in first["messages"][0]["text"]
    assert [m["role"] for m in last["messages"]] == ["user"] + ["assistant", "tool_results"] * 4
    assert {t["name"] for t in first["tools"]} == {
        "define_signal", "finish", "inspect_results", "query_events", "record_hypothesis",
        "report_finding", "run_backtest", "view_notebook",
    }


def test_step_cap_ends_a_looping_script(fake_session):
    s = fake_session("R2", max_steps=3)
    claims = run_session(s, FakeLLM([turn(HYPOTHESIS)] * 10))
    assert claims == []
    assert end_event(s)["reason"] == "step_cap" and s.step == 3


def test_invalid_tool_call_is_returned_and_the_loop_continues(fake_session):
    s = fake_session("R2")
    llm = FakeLLM([turn(("buy_stock", {"ticker": "X"})), turn(FINISH)])
    run_session(s, llm)
    results = llm.calls[1]["messages"][-1]
    assert results["role"] == "tool_results"
    assert results["results"][0]["is_error"]
    assert results["results"][0]["content"].startswith("Unknown tool 'buy_stock'")
    assert end_event(s)["reason"] == "finished" and s.invalid_calls == 1


def test_r2_cannot_call_search_announcements(fake_session):
    s = fake_session("R2")
    llm = FakeLLM([turn(("search_announcements", {"query": "buyback"})), turn(FINISH)])
    run_session(s, llm)
    result = llm.calls[1]["messages"][-1]["results"][0]
    assert result["is_error"] and "Available tools:" in result["content"]
    assert "search_announcements" not in result["content"].split("Available tools:")[1]


def test_several_calls_in_one_turn_run_in_order(fake_session):
    s = fake_session("R2")
    llm = FakeLLM([
        turn(("define_signal", {"spec": SPEC}), ("run_backtest", {"signal_id": "sig-1"})),
        turn(FINISH),
    ])
    run_session(s, llm)
    results = llm.calls[1]["messages"][-1]["results"]
    assert [r["tool_call_id"] for r in results] == ["t0", "t1"]
    assert not any(r["is_error"] for r in results)
    assert s.tested_signals == {"sig-1"}


def test_token_cap(fake_session):
    s = fake_session("R2", max_session_input_tokens=400_000)
    run_session(s, FakeLLM([turn(HYPOTHESIS, input_tokens=150_000)] * 5))
    end = end_event(s)
    assert end["reason"] == "token_cap"
    assert end["totals"]["steps"] == 3 and end["totals"]["input_tokens"] == 450_000


def test_token_cap_counts_cached_input(fake_session):
    s = fake_session("R2", max_session_input_tokens=400_000)
    cached = turn(HYPOTHESIS, input_tokens=1_000)
    cached.usage.cache_read_input_tokens = 199_000
    run_session(s, FakeLLM([cached] * 5))
    assert end_event(s)["reason"] == "token_cap" and s.step == 2


def test_nudge_then_end_without_tools(fake_session):
    s = fake_session("R2")
    llm = FakeLLM([turn(text="I think I am done."), turn(text="Really done.")])
    run_session(s, llm)
    end = end_event(s)
    assert end["reason"] == "no_tool_calls" and end["totals"]["nudges"] == 1
    assert llm.calls[1]["messages"][-1] == {"role": "user", "text": NUDGE}
    assert [e["kind"] for e in s.trace.events].count("nudge") == 1


def test_nudge_then_resume(fake_session):
    s = fake_session("R2")
    run_session(s, FakeLLM([turn(text="Thinking."), turn(FINISH)]))
    end = end_event(s)
    assert end["reason"] == "finished" and end["totals"]["nudges"] == 1


def test_only_one_nudge_per_session(fake_session):
    s = fake_session("R2")
    run_session(s, FakeLLM([turn(text="a"), turn(HYPOTHESIS), turn(text="b")]))
    end = end_event(s)
    assert end["reason"] == "no_tool_calls" and end["totals"]["nudges"] == 1


def test_refusal_ends_immediately_without_nudge(fake_session):
    s = fake_session("R2")
    llm = FakeLLM([turn(stop_reason="refusal"), turn(FINISH)])
    run_session(s, llm)
    end = end_event(s)
    assert end["reason"] == "refusal" and end["totals"]["nudges"] == 0
    assert len(llm.calls) == 1


def test_client_error_closes_the_trace_and_reraises(fake_session):
    s = fake_session("R2")
    with pytest.raises(RuntimeError, match="responses"):
        run_session(s, FakeLLM([turn(HYPOTHESIS)]))  # script runs out on turn 2
    assert end_event(s)["reason"] == "error"
