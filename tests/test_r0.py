import pandas as pd
import pytest

from baselines.r0 import R0System
from evaluation.systems import UniverseView


def empty_view():
    return UniverseView(
        universe_id="test",
        prices=pd.DataFrame(),
        benchmarks=pd.DataFrame(),
        events=pd.DataFrame(),
        research_start=None,
        research_end=None,
    )


def test_select_dividend_increases():
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d", "e"],
            "event_type": [
                "dividend",
                "dividend",
                "dividend",
                "buyback",
                "dividend",
            ],
            "dividend_change_pct": [15.0, 10.0, -5.0, 20.0, None],
        }
    )
    original = events.copy(deep=True)

    selected = R0System().select_dividend_increases(events)

    assert selected["event_id"].tolist() == ["a"]
    pd.testing.assert_frame_equal(events, original)


def test_calculate_dividend_returns():
    dates = pd.bdate_range("2020-01-06", periods=6)

    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["ABC"] * 6,
            "ret": [0.01] * 5 + [0.50],
            "is_bad_price": [False] * 6,
        }
    )
    benchmarks = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * 6,
            "ret": [0.002] * 6,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["complete", "too_late"],
            "ticker": ["ABC", "ABC"],
            "first_tradeable_date": [dates[0], dates[-1]],
            "event_type": ["dividend", "dividend"],
            "dividend_change_pct": [15.0, 15.0],
        }
    )
    view = UniverseView(
        universe_id="test",
        prices=prices,
        benchmarks=benchmarks,
        events=events,
        research_start=dates[0].date(),
        research_end=dates[-1].date(),
    )

    cars = R0System().calculate_dividend_returns(view)

    assert cars["event_id"].tolist() == ["complete"]
    assert cars.iloc[0]["car"] == pytest.approx(0.04)
    assert cars.iloc[0]["n_days"] == 5

    skipped = cars.attrs["skipped"]
    assert skipped["event_id"].tolist() == ["too_late"]
    assert skipped.iloc[0]["reason"] == "past_data"


def test_summarize_positive_returns():
    cars = pd.DataFrame({"car": [0.01, 0.03] * 10})

    result = R0System().summarize_returns(cars)

    assert result["status"] == "tested"
    assert result["n_events"] == 20
    assert result["mean_car"] == pytest.approx(0.02)
    assert result["t_stat"] == pytest.approx(76**0.5)
    assert 0 < result["p_value_raw"] < 0.05


def test_summarize_insufficient_events():
    cars = pd.DataFrame({"car": [0.01, 0.03] * 9})

    result = R0System().summarize_returns(cars)

    assert result["status"] == "insufficient_events"
    assert result["n_events"] == 18
    assert result["p_value_raw"] is None


def test_summarize_zero_mean_returns():
    cars = pd.DataFrame({"car": [-0.01, 0.01] * 10})

    result = R0System().summarize_returns(cars)

    assert result["status"] == "tested"
    assert result["mean_car"] == pytest.approx(0.0)
    assert result["p_value_raw"] == pytest.approx(1.0)


def test_summarize_identical_returns():
    cars = pd.DataFrame({"car": [0.02] * 20})

    result = R0System().summarize_returns(cars)

    assert result["status"] == "zero_variance"
    assert result["p_value_raw"] is None


def test_run_returns_a_finding(monkeypatch):
    system = R0System()
    cars = pd.DataFrame(
        {
            "event_id": [f"event-{i}" for i in range(20)],
            "car": [0.01, 0.03] * 10,
        }
    )
    monkeypatch.setattr(
        system,
        "calculate_dividend_returns",
        lambda view: cars,
    )

    result = system.run(empty_view(), budget=1, seed=0)

    assert result.metadata["backtests_used"] == 1
    assert len(result.findings) == 1

    finding = result.findings[0]
    assert finding.system == "R0"
    assert finding.n_events == 20
    assert finding.effect_car == pytest.approx(0.02)
    assert finding.passed_correction
    assert finding.p_value_corrected == finding.p_value_raw


def test_zero_budget_does_not_calculate_returns(monkeypatch):
    system = R0System()

    def unexpected_calculation(view):
        raise AssertionError("No calculation allowed with zero budget")

    monkeypatch.setattr(
        system,
        "calculate_dividend_returns",
        unexpected_calculation,
    )

    result = system.run(empty_view(), budget=0, seed=0)

    assert result.findings == []
    assert result.metadata["backtests_used"] == 0


def test_holm_adjustment():
    from baselines.r0 import holm_adjust

    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    assert holm_adjust([0.02]) == pytest.approx([0.02])


def test_grid_and_budget(monkeypatch):
    system = R0System()
    cars = pd.DataFrame(
        {
            "event_id": [f"event-{i}" for i in range(20)],
            "car": [0.01, 0.03] * 10,
        }
    )
    monkeypatch.setattr(system, "calculate_dividend_returns", lambda view: cars)
    monkeypatch.setattr(system, "calculate_other_returns", lambda *args: cars)

    result = system.run(empty_view(), budget=2, seed=0)

    assert result.metadata["backtests_used"] == 2
    assert [h["hypothesis_id"] for h in result.metadata["hypotheses"]] == [
        "dividend_increase_gt10",
        "special_dividend",
    ]

    complete = system.run(empty_view(), budget=30, seed=0)

    assert complete.metadata["backtests_used"] == 3
    assert len(complete.findings) == 3
    assert complete == system.run(empty_view(), budget=30, seed=0)

    raw = complete.findings[0].p_value_raw
    for finding in complete.findings:
        assert finding.p_value_corrected == pytest.approx(min(1.0, 3 * raw))


def test_insufficient_attempt_still_counts(monkeypatch):
    system = R0System()
    cars = pd.DataFrame({"event_id": ["one"], "car": [0.02]})
    monkeypatch.setattr(system, "calculate_dividend_returns", lambda view: cars)

    result = system.run(empty_view(), budget=1, seed=0)

    assert result.findings == []
    assert result.metadata["backtests_used"] == 1
    hypothesis = result.metadata["hypotheses"][0]
    assert hypothesis["status"] == "insufficient_events"
    assert not hypothesis["passed_correction"]


def test_r0_registration():
    from evaluation.systems import ResearchSystem, make_systems

    system = make_systems(["r0"], holding_days=3, min_events=25)[0]

    assert isinstance(system, ResearchSystem)
    assert system.holding_days == 3
    assert system.min_events == 25
