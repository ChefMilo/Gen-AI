"""Scripted agent -> real engine -> budget rejection -> finalized finding.

No model API calls or Git provenance commands are used in these tests.
"""

import json

import pytest

import agent.session as session_module
import agent.tools as tools
from agent.llm import FakeLLM, LLMResponse, ToolCall, Usage
from agent.loop import run_session
from agent.session import new_session
from harness.agent_adapter import finalize_agent_session, register_backtest_tools
from harness.backtester import GuardrailError, HarnessConfig, ResearchBacktester
from test_backtest_engine import sample_data as sample_data
from test_harness import harness_data as harness_data


class ControlledEventPort:
    """Minimal test-only port; does not claim to verify query/retrieval isolation."""

    def __init__(self, event_ids):
        self._ids = set(event_ids)

    def available_fields(self):
        return {}

    def existing_event_ids(self, event_ids):
        return self._ids.intersection(event_ids)

    def query(self, event_type, conditions, limit):
        raise AssertionError("This script must not use the event-query tool")


def turn(index, name, arguments, input_tokens=0):
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call-{index}", name=name, arguments=arguments)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=input_tokens),
    )


@pytest.fixture
def real_session(harness_data, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_TOOLS", dict(tools._TOOLS))
    # new_session normally calls Git through this provenance hook.
    monkeypatch.setattr(session_module, "code_version", lambda: None)
    register_backtest_tools()
    prices, benchmarks, events = harness_data
    backtester = ResearchBacktester(
        prices,
        benchmarks,
        events,
        HarnessConfig(research_start="2020-01-06", research_end="2020-01-13", budget=1),
        tmp_path / "agent-audit.jsonl",
    )
    return new_session(
        system="R2",
        config={
            "research_start": "2020-01-06",
            "research_end": "2020-01-13",
            "min_events": 20,
            "agent": {
                "model": "scripted",
                "max_tokens": 1000,
                "max_steps": 10,
                "tool_result_max_chars": 4000,
            },
        },
        universe_id="controlled-engine-test",
        backtester=backtester,
        events=ControlledEventPort(events["event_id"]),
        in_memory=True,
    )


def test_scripted_agent_uses_real_harness(real_session, tmp_path):
    session = real_session
    script = [
        turn(
            1,
            "record_hypothesis",
            {
                "hypothesis": "Dividend announcement effect",
                "rationale": "Controlled integration example",
            },
        ),
        turn(
            2,
            "define_signal",
            {
                "spec": {"event_type": "dividend", "holding_days": 5},
                "hypothesis_id": "h1",
            },
        ),
        turn(3, "run_backtest", {"signal_id": "sig-1"}),
        turn(4, "run_backtest", {"signal_id": "sig-1"}),
        turn(5, "inspect_results", {"signal_id": "sig-1", "by": "year"}),
        turn(
            6,
            "report_finding",
            {
                "signal_id": "sig-1",
                "rationale": "Known positive effect in test data",
                "evidence": [],
            },
        ),
        turn(7, "finish", {"reason": "Test completed"}),
    ]
    run_session(session, FakeLLM(script))
    findings = finalize_agent_session(session)
    assert session.done and session.invalid_calls == 1
    assert session.backtester.budget_status()["tests_used"] == 1
    assert len(findings) == 1
    finding = findings[0]
    assert finding.system == "R2" and finding.n_events == 20
    assert finding.effect_car == pytest.approx(0.02)
    assert finding.passed_correction and finding.signal_spec.holding_days == 5
    assert session.signal_event_ids["sig-1"] == [f"event-{i}" for i in range(20)]
    audit = [
        json.loads(line)
        for line in (tmp_path / "agent-audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        entry["action"] == "run_rejected" and entry["reason"] == "budget_exhausted"
        for entry in audit
    )
    assert audit[-1]["action"] == "session_finalized"
    with pytest.raises(GuardrailError, match="closed"):
        session.backtester.run("sig-1")


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("run_backtest", {"signal_id": "invented"}),
        (
            "define_signal",
            {
                "spec": {
                    "event_type": "dividend",
                    "text_filter": {"feature": "dividend_framing", "equals": "confidence"},
                }
            },
        ),
        ("define_signal", {"spec": {"event_type": "dividend", "holding_days": 0}}),
    ],
)
def test_rejected_tools_are_recoverable(real_session, name, arguments):
    run_session(
        real_session,
        FakeLLM(
            [
                turn(1, name, arguments),
                turn(2, "finish", {"reason": "Recovered"}),
            ]
        ),
    )
    assert real_session.done and real_session.invalid_calls == 1
    assert real_session.backtester.budget_status()["tests_used"] == 0
    assert finalize_agent_session(real_session) == []


def test_internal_errors_are_not_disguised_as_tool_rejections(real_session, monkeypatch):
    signal_id = real_session.backtester.define({"event_type": "dividend"})

    def broken_run(signal_id):
        raise RuntimeError("controlled internal failure")

    monkeypatch.setattr(real_session.backtester, "run", broken_run)
    with pytest.raises(RuntimeError, match="controlled internal failure"):
        run_session(
            real_session,
            FakeLLM(
                [
                    turn(1, "run_backtest", {"signal_id": signal_id}),
                ]
            ),
        )
    assert real_session.trace.events[-1]["reason"] == "error"
    with pytest.raises(GuardrailError, match="ended with an error"):
        finalize_agent_session(real_session)


def test_cannot_finalize_before_loop_ends(real_session):
    with pytest.raises(GuardrailError, match="Finish the agent loop"):
        finalize_agent_session(real_session)


@pytest.mark.parametrize("cap", ["step", "input_tokens"])
def test_resource_caps_end_scripted_loop(real_session, cap):
    if cap == "step":
        real_session.config["agent"]["max_steps"] = 1
        expected = "step_cap"
    else:
        real_session.config["agent"]["max_session_input_tokens"] = 1
        expected = "token_cap"
    run_session(
        real_session,
        FakeLLM(
            [
                turn(
                    1,
                    "record_hypothesis",
                    {
                        "hypothesis": "An unfinished idea",
                        "rationale": "Resource cap test",
                    },
                    input_tokens=10,
                ),
            ]
        ),
    )
    assert real_session.trace.events[-1]["reason"] == expected
    assert finalize_agent_session(real_session) == []


def test_model_receives_summary_not_full_returns(real_session):
    signal_id = real_session.backtester.define({"event_type": "dividend"})
    content, is_error = tools.dispatch(real_session, "run_backtest", {"signal_id": signal_id})
    assert not is_error
    view = json.loads(content)
    assert view["status"] == "tested" and view["n_events"] == 20
    assert len(view["example_event_ids"]) == 3
    assert "event_returns" not in view and "event_ids" not in view
    assert len(real_session.signal_event_ids[signal_id]) == 20
