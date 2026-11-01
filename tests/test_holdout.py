import json

import pytest
from pydantic import ValidationError

from harness.backtester import HarnessConfig, ResearchBacktester
from harness.holdout import HoldoutConfig, evaluate_holdout_once
from test_backtest_engine import sample_data as sample_data
from test_harness import harness_data as harness_data


def discovery(data, tmp_path):
    backtester = ResearchBacktester(
        *data,
        HarnessConfig(research_start="2020-01-06", research_end="2020-01-13", budget=1),
        tmp_path / "discovery.jsonl",
    )
    signal = backtester.define({"event_type": "dividend"})
    backtester.run(signal)
    return backtester.finalize(
        [{"signal_id": signal, "rationale": "Synthetic known effect"}], "research", "R2"
    )


def test_holdout_freezes_before_loading_and_runs_once(harness_data, tmp_path):
    findings = discovery(harness_data, tmp_path)
    # The synthetic discovery above supplies a known result. These dates are
    # solely a unit-test fixture, not an actual research/holdout study.
    config = HoldoutConfig(
        research_end="2019-12-31", holdout_start="2020-01-06", holdout_end="2020-01-13"
    )
    ledger = tmp_path / "holdout.json"
    calls = []

    def loader(start, end):
        assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "reserved"
        calls.append((start, end))
        return harness_data

    report = evaluate_holdout_once(findings, config, ledger, loader)
    assert report["status"] == "completed" and len(calls) == 1
    assert report["results"][0]["retention_ratio"] == pytest.approx(1)
    assert report["results"][0]["passed_correction"]
    with pytest.raises(FileExistsError):
        evaluate_holdout_once(findings, config, ledger, loader)
    assert len(calls) == 1


def test_failed_holdout_remains_reserved_against_retry(harness_data, tmp_path):
    findings = discovery(harness_data, tmp_path)
    config = HoldoutConfig(
        research_end="2019-12-31", holdout_start="2020-01-06", holdout_end="2020-01-13"
    )
    ledger = tmp_path / "holdout.json"

    def fail(*args):
        raise RuntimeError("controlled data failure")

    with pytest.raises(RuntimeError, match="controlled data failure"):
        evaluate_holdout_once(findings, config, ledger, fail)
    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(FileExistsError):
        evaluate_holdout_once(findings, config, ledger, fail)


def test_overlapping_periods_are_rejected():
    with pytest.raises(ValidationError, match="no overlap"):
        HoldoutConfig(
            research_end="2020-01-10", holdout_start="2020-01-10", holdout_end="2020-01-13"
        )
