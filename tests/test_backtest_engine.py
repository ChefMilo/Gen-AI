import pandas as pd
import pytest

from backtest.engine import run_event_study
from dsl.signal import SignalSpec


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2020-01-06", periods=6)
    price_rows = []
    event_rows = []

    for i in range(20):
        ticker = f"T{i:02d}"
        car = 0.01 if i % 2 == 0 else 0.03

        for day in dates:
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "ret": 0.002 + car / 5,
                    "is_bad_price": False,
                }
            )

        event_rows.append(
            {
                "event_id": f"event-{i}",
                "ticker": ticker,
                "event_type": "dividend",
                "first_tradeable_date": dates[0],
                "is_new_announcement": True,
                "extraction_status": "passed",
            }
        )

    benchmark = pd.DataFrame(
        {
            "ticker": ["SPY"] * 6,
            "date": dates,
            "ret": [0.002] * 6,
        }
    )

    return (
        pd.DataFrame(price_rows),
        benchmark,
        pd.DataFrame(event_rows),
    )


def run_sample(sample_data, **kwargs):
    prices, benchmark, events = sample_data
    spec = SignalSpec(event_type="dividend", **kwargs)

    return run_event_study(prices, benchmark, events, spec, "sig-test")


def test_known_returns(sample_data):
    result = run_sample(sample_data)

    assert result.status == "tested"
    assert result.n_selected == 20
    assert result.n_events == 20
    assert result.mean_car == pytest.approx(0.02)
    assert result.t_stat == pytest.approx(76**0.5)
    assert result.p_value_raw < 0.05
    assert result.hit_rate == 1.0
    assert result.skipped_events == []


def test_short_preserves_raw_effect(sample_data):
    result = run_sample(sample_data, direction="short")

    assert result.mean_car == pytest.approx(0.02)
    assert result.directional_mean_car == pytest.approx(-0.02)
    assert result.hit_rate == 0.0


def test_insufficient_events(sample_data):
    prices, benchmark, events = sample_data

    result = run_event_study(
        prices,
        benchmark,
        events.iloc[:19],
        SignalSpec(event_type="dividend"),
        "sig-small",
    )

    assert result.status == "insufficient_events"
    assert result.n_events == 19
    assert result.p_value_raw is None


def test_missing_stock_day_is_skipped(sample_data):
    prices, benchmark, events = sample_data
    missing = prices["ticker"].eq("T00") & prices["date"].eq(benchmark["date"].iloc[1])
    prices = prices.loc[~missing].copy()

    result = run_sample((prices, benchmark, events))

    assert result.n_events == 19
    assert result.skipped_events[0].event_id == "event-0"
    assert result.skipped_events[0].reason == "missing_trading_day"


def test_nonfinite_return_is_skipped(sample_data):
    prices, benchmark, events = sample_data
    prices = prices.copy()
    prices.loc[0, "ret"] = float("nan")

    result = run_sample((prices, benchmark, events))

    assert result.n_events == 19
    assert result.skipped_events[0].reason == "nonfinite_return"


def test_requested_benchmark_is_used(sample_data):
    prices, benchmark, events = sample_data
    other = benchmark.assign(ticker="IWM", ret=0.0)
    combined = pd.concat([benchmark, other], ignore_index=True)

    result = run_sample(
        (prices, combined, events),
        benchmark="IWM",
    )

    assert result.mean_car == pytest.approx(0.03)


def test_inputs_unchanged_and_output_serializable(sample_data):
    originals = [frame.copy(deep=True) for frame in sample_data]

    result = run_sample(sample_data)

    for actual, original in zip(sample_data, originals, strict=True):
        pd.testing.assert_frame_equal(actual, original)

    assert type(result).model_validate_json(result.model_dump_json()) == result
