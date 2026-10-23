import json

from agent.llm import LLMResponse, ToolCall, Usage
from agent.trace import Trace, read_trace


def turn(stop_reason="tool_use", inp=100, out=20, thinking=""):
    return LLMResponse(
        text="Let me test.",
        thinking=thinking,
        tool_calls=[ToolCall(id="t1", name="run_backtest", arguments={"signal_id": "s1"})],
        stop_reason=stop_reason,
        usage=Usage(input_tokens=inp, output_tokens=out),
        raw=[{"type": "thinking", "thinking": thinking, "signature": "SECRET-SIG"}],
    )


def write_session(trace: Trace) -> None:
    trace.session_start({"agent": {"max_steps": 60}}, "SYSTEM", "fake-0", "m", 16000, None)
    trace.model_turn(1, turn(thinking="Small caps first."))
    trace.tool_call(1, "run_backtest", {"signal_id": "s1"}, '{"ok": 1}', False, 12.34)
    trace.model_turn(2, turn(inp=150, out=30))
    trace.tool_call(2, "nope", {}, "unknown tool", True, 0.1)
    trace.model_turn(3, turn(stop_reason="end_turn", inp=200, out=5))
    trace.session_end("no_tool_calls")


def test_jsonl_lines_are_valid_ordered_and_tagged(tmp_path):
    trace = Trace("sess-1", tmp_path)
    write_session(trace)
    path = tmp_path / "sess-1.jsonl"
    events = read_trace(path)
    assert events == json.loads(json.dumps(trace.events, default=str))
    assert [e["kind"] for e in events] == [
        "session_start", "model_turn", "tool_call", "model_turn", "tool_call", "model_turn",
        "session_end",
    ]
    assert [e["seq"] for e in events] == list(range(7))
    assert all(e["session_id"] == "sess-1" and e["ts"] for e in events)
    assert [e["ts"] for e in events] == sorted(e["ts"] for e in events)


def test_session_start_records_effective_settings():
    trace = Trace("s")
    event = trace.session_start({}, "SYSTEM", "fake-0", "m", 16000, "high")
    assert event["settings"] == {"model": "m", "max_tokens": 16000, "effort": "high"}
    assert event["system"] == "SYSTEM" and event["universe_id"] == "fake-0"


def test_model_turn_keeps_thinking_text_but_never_raw_blocks(tmp_path):
    trace = Trace("s", tmp_path)
    event = trace.model_turn(1, turn(thinking="Small caps first."))
    assert event["thinking"] == "Small caps first."
    assert event["stop_reason"] == "tool_use"
    assert event["tool_calls"] == [
        {"id": "t1", "name": "run_backtest", "arguments": {"signal_id": "s1"}}
    ]
    line = (tmp_path / "s.jsonl").read_text(encoding="utf-8")
    assert "raw" not in json.loads(line) and "SECRET-SIG" not in line


def test_totals():
    trace = Trace("s")
    write_session(trace)
    expected = {
        "steps": 3,
        "tool_calls": 2,
        "invalid_calls": 1,
        "input_tokens": 450,
        "output_tokens": 55,
        "stop_reasons": {"tool_use": 2, "end_turn": 1},
    }
    assert trace.totals() == expected
    assert trace.events[-1]["totals"] == expected
    assert trace.events[-1]["reason"] == "no_tool_calls"


def test_totals_count_every_stop_reason():
    trace = Trace("s")
    for reason in ["tool_use", "max_tokens", "refusal", "max_tokens"]:
        trace.model_turn(1, turn(stop_reason=reason))
    assert trace.totals()["stop_reasons"] == {"tool_use": 1, "max_tokens": 2, "refusal": 1}


def test_in_memory_trace_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = Trace("s")
    write_session(trace)
    assert trace.path is None and list(tmp_path.iterdir()) == []
