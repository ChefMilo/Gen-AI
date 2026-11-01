import pytest

from harness.backtester import (
    GuardrailError,
    HarnessConfig,
    ResearchBacktester,
)
from test_backtest_engine import sample_data as sample_data
from test_harness import harness_data as harness_data


def make_session(
    data,
    tmp_path,
    *,
    mode="guarded",
    budget=1,
    end="2020-01-13",
):
    prices, benchmarks, events = data

    return ResearchBacktester(
        prices,
        benchmarks,
        events,
        HarnessConfig(
            research_start="2020-01-06",
            research_end=end,
            mode=mode,
            budget=budget,
        ),
        tmp_path / f"{mode}.jsonl",
    )


def test_unguarded_can_exceed_backtest_budget(
    harness_data,
    tmp_path,
):
    harness = make_session(
        harness_data,
        tmp_path,
        mode="unguarded",
    )
    signal_id = harness.define({"event_type": "dividend"})

    harness.run(signal_id)
    result = harness.run(signal_id)

    assert result["tests_used"] == 2
    assert result["budget_remaining"] is None
    assert not result["budget_enforced"]


@pytest.mark.parametrize("mode", ["guarded", "unguarded"])
def test_both_modes_keep_the_time_lockout(
    harness_data,
    tmp_path,
    mode,
):
    harness = make_session(
        harness_data,
        tmp_path,
        mode=mode,
        end="2020-01-08",
    )
    signal_id = harness.define({"event_type": "dividend"})

    result = harness.run(signal_id)

    assert result["n_events"] == 0
    assert result["status"] == "insufficient_events"


def test_guarded_and_unguarded_correction(
    harness_data,
    tmp_path,
    monkeypatch,
):
    import harness.backtester as module

    original = module.run_event_study

    def controlled_result(*args, **kwargs):
        result = original(*args, **kwargs)
        result.p_value_raw = 0.04
        return result

    monkeypatch.setattr(
        module, "run_event_study", controlled_result
    )

    outcomes = {}

    for mode, system in [
        ("guarded", "R3"),
        ("unguarded", "R3-U"),
    ]:
        harness = make_session(
            harness_data,
            tmp_path,
            mode=mode,
            budget=2,
        )
        first = harness.define({"event_type": "dividend"})
        second = harness.define({"event_type": "dividend"})

        harness.run(first)
        harness.run(second)

        findings = harness.finalize(
            claims=[
                {
                    "signal_id": first,
                    "rationale": "Controlled comparison",
                }
            ],
            session_id=mode,
            system=system,
        )
        outcomes[mode] = findings[0]

    assert outcomes["guarded"].p_value_corrected == pytest.approx(
        0.08
    )
    assert not outcomes["guarded"].passed_correction

    assert outcomes["unguarded"].p_value_corrected == pytest.approx(
        0.04
    )
    assert outcomes["unguarded"].passed_correction


def test_unguarded_cannot_be_labelled_r3(
    harness_data,
    tmp_path,
):
    harness = make_session(
        harness_data,
        tmp_path,
        mode="unguarded",
    )

    with pytest.raises(GuardrailError, match="R3-U"):
        harness.finalize(
            claims=[],
            session_id="wrong-label",
            system="R3",
        )