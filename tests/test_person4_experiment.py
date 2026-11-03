import json

import pandas as pd
import pytest

import agent.tools as tools
from agent.llm import FakeLLM, LLMResponse, ToolCall
from harness.agent_adapter import finalize_agent_session, register_research_tools
from harness.agent_session import make_research_session
from harness.backtester import HarnessConfig, ResearchBacktester
from harness.limits import ModelLimits, ResourceLimitedClient
from harness.offline import OfflineConfig, run_offline
from test_backtest_engine import sample_data as sample_data
from test_harness import harness_data as harness_data


def test_resource_cap_is_a_normal_loop_termination(harness_data, tmp_path, monkeypatch):
    from agent.loop import run_session

    monkeypatch.setattr(tools, "_TOOLS", dict(tools._TOOLS))
    backtester = ResearchBacktester(
        *harness_data,
        HarnessConfig(research_start="2020-01-06", research_end="2020-01-13", budget=1),
        tmp_path / "cap-audit.jsonl",
    )
    session = make_research_session(backtester, system="R2", universe_id="cap")
    response = LLMResponse(
        tool_calls=[
            ToolCall(
                id="h",
                name="record_hypothesis",
                arguments={"hypothesis": "A bounded hypothesis", "rationale": "Resource cap test"},
            )
        ],
        stop_reason="tool_use",
    )
    client = ResourceLimitedClient(FakeLLM([response]), ModelLimits(max_calls=1), lambda *args: 0)
    register_research_tools()
    run_session(session, client)
    assert session.trace.events[-1]["reason"] == "model_call_cap"
    assert finalize_agent_session(session) == []
    assert client.calls == 1


def test_offline_pipeline_is_paired_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_TOOLS", dict(tools._TOOLS))
    config = OfflineConfig(n_planted=1, n_placebo=1, n_boot=20)
    first = run_offline(config, tmp_path / "first")
    second = run_offline(config, tmp_path / "second")
    assert first["paid_model_calls"] == second["paid_model_calls"] == 0
    assert first["paired_sessions"] == 4
    a = pd.read_csv(tmp_path / "first/per_universe.csv")
    b = pd.read_csv(tmp_path / "second/per_universe.csv")
    pd.testing.assert_frame_equal(a, b)
    assert set(a["system"]) == {"R3", "R3-U"}
    assert set(a["universe_type"]) == {"planted", "placebo"}
    assert (a.groupby("universe_id")["system"].nunique() == 2).all()
    assert (a.loc[a["system"].eq("R3"), "backtests_used"] == 3).all()
    assert (a.loc[a["system"].eq("R3-U"), "backtests_used"] == 8).all()
    runs = list((tmp_path / "first/runs").glob("*/*/run.json"))
    assert len(runs) == 4
    for path in runs:
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["metadata"]["client_kind"] == "scripted"
        assert result["metadata"]["portfolio_reports"]
    assert "not evidence of real LLM" in (tmp_path / "first/REPORT.md").read_text(encoding="utf-8")


def test_offline_output_is_not_silently_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_offline(OfflineConfig(n_planted=0, n_placebo=1), output)
