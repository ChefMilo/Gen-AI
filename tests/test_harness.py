import json

import pandas as pd
import pytest

from harness.backtester import (
    GuardrailError,
    HarnessConfig,
    ResearchBacktester,
)
from test_backtest_engine import sample_data as sample_data


@pytest.fixture
def harness_data(sample_data):
    prices, benchmarks, events = sample_data
    events = events.copy()

    # The engine fixture starts on 6 January 2020.
    events["acceptance_datetime"] = pd.Timestamp("2020-01-06 09:00:00")
    return prices, benchmarks, events


def make_harness(
    data,
    tmp_path,
    *,
    budget=1,
    research_end="2020-01-13",
):
    prices, benchmarks, events = data

    return ResearchBacktester(
        prices,
        benchmarks,
        events,
        HarnessConfig(
            research_start="2020-01-06",
            research_end=research_end,
            budget=budget,
        ),
        audit_path=tmp_path / "audit.jsonl",
    )


def test_budget_is_enforced(harness_data, tmp_path):
    harness = make_harness(harness_data, tmp_path)
    signal_id = harness.define({"event_type": "dividend"})

    result = harness.run(signal_id)

    assert result["status"] == "tested"
    assert result["tests_used"] == 1
    assert result["budget_remaining"] == 0

    with pytest.raises(GuardrailError, match="exhausted"):
        harness.run(signal_id)


def test_cutoff_blocks_future_price_window(harness_data, tmp_path):
    harness = make_harness(
        harness_data,
        tmp_path,
        research_end="2020-01-08",
    )
    signal_id = harness.define({"event_type": "dividend"})

    result = harness.run(signal_id)

    assert result["status"] == "insufficient_events"
    assert result["n_events"] == 0
    assert result["tests_used"] == 1
    assert all(row["reason"] == "past_data" for row in result["skipped_events"])


def test_future_announcements_are_hidden(harness_data, tmp_path):
    prices, benchmarks, events = harness_data
    events = events.copy()
    events["acceptance_datetime"] = pd.Timestamp("2020-02-01 09:00:00")
    harness = make_harness(
        (prices, benchmarks, events),
        tmp_path,
    )
    signal_id = harness.define({"event_type": "dividend"})

    result = harness.run(signal_id)

    assert result["n_selected"] == 0
    assert result["n_events"] == 0


def test_unknown_signal_does_not_run(harness_data, tmp_path):
    harness = make_harness(harness_data, tmp_path)

    with pytest.raises(GuardrailError, match="Unknown"):
        harness.run("invented-signal")

    assert harness.budget_status()["tests_used"] == 0


def test_calculation_error_consumes_budget(
    harness_data,
    tmp_path,
    monkeypatch,
):
    import harness.backtester as module

    harness = make_harness(harness_data, tmp_path)
    signal_id = harness.define({"event_type": "dividend"})

    def broken_calculation(*args, **kwargs):
        raise ValueError("controlled failure")

    monkeypatch.setattr(
        module,
        "run_event_study",
        broken_calculation,
    )

    with pytest.raises(ValueError, match="controlled failure"):
        harness.run(signal_id)

    assert harness.budget_status()["tests_used"] == 1

    with pytest.raises(GuardrailError, match="exhausted"):
        harness.run(signal_id)


def test_audit_records_success_and_rejection(harness_data, tmp_path):
    harness = make_harness(harness_data, tmp_path)
    signal_id = harness.define({"event_type": "dividend"})
    harness.run(signal_id)

    with pytest.raises(GuardrailError):
        harness.run(signal_id)

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]

    assert [entry["action"] for entry in entries] == [
        "session_start",
        "signal_defined",
        "backtest_started",
        "backtest_completed",
        "run_rejected",
    ]
    assert [entry["sequence"] for entry in entries] == list(range(len(entries)))


def test_after_close_event_cannot_use_same_day(
    harness_data,
    tmp_path,
):
    prices, benchmarks, events = harness_data
    events = events.copy()
    events.loc[0, "acceptance_datetime"] = pd.Timestamp("2020-01-06 17:00:00")

    with pytest.raises(GuardrailError, match="Invalid first_tradeable_date"):
        make_harness(
            (prices, benchmarks, events),
            tmp_path,
        )


def test_signal_options_match_available_data(
    harness_data,
    tmp_path,
):
    harness = make_harness(harness_data, tmp_path)

    options = harness.signal_options()

    assert "SPY" in options["benchmarks"]
    assert "year" in options["breakdowns"]
    assert "sector" not in options["breakdowns"]
    assert "in" not in options["operators"]["number"]


def test_breakdown_does_not_spend_another_test(
    harness_data,
    tmp_path,
):
    harness = make_harness(harness_data, tmp_path)
    signal_id = harness.define({"event_type": "dividend"})
    harness.run(signal_id)

    result = harness.breakdown(signal_id, by="year")

    assert harness.budget_status()["tests_used"] == 1
    assert result["descriptive_only"]
    assert result["groups"]["2020"]["n_events"] == 20
    assert result["groups"]["2020"]["mean_car"] == pytest.approx(0.02)
    assert "p_value_raw" not in result["groups"]["2020"]


def test_breakdown_requires_completed_backtest(
    harness_data,
    tmp_path,
):
    harness = make_harness(harness_data, tmp_path)
    signal_id = harness.define({"event_type": "dividend"})

    with pytest.raises(GuardrailError, match="Run the signal"):
        harness.breakdown(signal_id, by="year")
