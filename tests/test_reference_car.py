"""The vectorised event_cars against the original per-event implementation (the oracle)."""

import numpy as np
import pandas as pd
import pytest

from evaluation.reference_car import CAR_COLUMNS, SKIPPED_COLUMNS, event_cars
from store.schema import DEFAULT_BENCHMARK
from universes.fake_events import make_fake_events
from universes.planted import eligible
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"


def _as_timestamps(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values)).dt.normalize()


def oracle_event_cars(
    prices: pd.DataFrame, benchmarks: pd.DataFrame, events: pd.DataFrame, holding_days: int
) -> pd.DataFrame:
    """event_cars as first written: one iterrows pass over the events."""
    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")
    bench = benchmarks[benchmarks["ticker"] == DEFAULT_BENCHMARK]
    bench_ret = pd.Series(bench["ret"].to_numpy(), index=_as_timestamps(bench["date"]).to_numpy())

    by_ticker = {}
    for ticker, rows in prices.groupby("ticker", sort=False):
        rows = rows.assign(date=_as_timestamps(rows["date"]).to_numpy()).sort_values("date")
        by_ticker[ticker] = (
            rows["date"].to_numpy(),
            rows["ret"].to_numpy(dtype=float),
            rows["is_bad_price"].to_numpy(dtype=bool),
        )

    cars, skipped = [], []
    day0s = _as_timestamps(events["first_tradeable_date"]).to_numpy()
    for (_, event), day0 in zip(events.iterrows(), day0s, strict=True):
        key = {
            "event_id": event["event_id"],
            "ticker": event["ticker"],
            "first_tradeable_date": event["first_tradeable_date"],
        }
        dates, ret, bad = by_ticker.get(event["ticker"], (np.array([], "datetime64[ns]"),) * 3)
        pos = int(np.searchsorted(dates, day0))
        if pos >= len(dates) or dates[pos] != day0:
            skipped.append({**key, "reason": "no_day0"})
            continue
        end = pos + holding_days
        if end > len(dates):
            skipped.append({**key, "reason": "past_data"})
            continue
        if bad[pos:end].any():
            skipped.append({**key, "reason": "bad_price"})
            continue
        spy = bench_ret.reindex(dates[pos:end]).to_numpy(dtype=float)
        if np.isnan(spy).any():
            skipped.append({**key, "reason": "no_benchmark"})
            continue
        cars.append({**key, "car": float(np.sum(ret[pos:end] - spy)), "n_days": holding_days})

    out = pd.DataFrame(cars, columns=CAR_COLUMNS)
    out.attrs["skipped"] = pd.DataFrame(skipped, columns=SKIPPED_COLUMNS)
    return out


def assert_same(prices, benchmarks, events, holding_days):
    got = event_cars(prices, benchmarks, events, holding_days)
    want = oracle_event_cars(prices, benchmarks, events, holding_days)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    pd.testing.assert_frame_equal(got.attrs["skipped"], want.attrs["skipped"], check_exact=True)
    return got


@pytest.fixture(scope="module")
def prices():
    return make_synthetic_prices(TICKERS, START, END, seed=0)


@pytest.fixture(scope="module")
def benchmarks():
    return make_synthetic_benchmarks(START, END, seed=0)


@pytest.fixture(scope="module")
def events(prices):
    return make_fake_events(prices, seed=0)


@pytest.mark.parametrize("holding_days", [1, 5, 20])
def test_matches_oracle_on_fake_data(prices, benchmarks, events, holding_days):
    cars = assert_same(prices, benchmarks, events, holding_days)
    assert len(cars) > 0


def test_matches_oracle_on_every_skip_reason(prices, benchmarks, events):
    rows = eligible(events).sample(400, random_state=0)
    # no_day0: a weekend, and a ticker with no prices.
    weekend = rows.iloc[[0]].assign(event_id="weekend",
                                    first_tradeable_date=pd.Timestamp("2016-01-09").date())
    unknown = rows.iloc[[1]].assign(event_id="unknown", ticker="ZZZ")
    # past_data: day 0 two rows before the end of a ticker's data.
    last = prices[prices["ticker"] == "T00"].sort_values("date")["date"].iloc[-2]
    late = rows.iloc[[2]].assign(event_id="late", ticker="T00",
                                 first_tradeable_date=last.date())
    events = pd.concat([rows, weekend, unknown, late], ignore_index=True)

    # bad_price on a window day; SPY missing on another event's window day.
    bad = prices.copy()
    t, d = rows.iloc[3]["ticker"], pd.Timestamp(rows.iloc[3]["first_tradeable_date"])
    later = bad[(bad["ticker"] == t) & (bad["date"] > d)].sort_values("date").index[1]
    bad.loc[later, "is_bad_price"] = True
    gap = pd.Timestamp(rows.iloc[4]["first_tradeable_date"]) + pd.offsets.BDay(1)
    bench = benchmarks[~((benchmarks["ticker"] == "SPY") & (benchmarks["date"] == gap))]
    # A NaN stock return is kept as a NaN CAR, not skipped.
    nan_ret = bad.copy()
    t5, d5 = rows.iloc[5]["ticker"], pd.Timestamp(rows.iloc[5]["first_tradeable_date"])
    nan_ret.loc[(nan_ret["ticker"] == t5) & (nan_ret["date"] == d5), "ret"] = np.nan

    cars = assert_same(nan_ret, bench, events, 5)
    reasons = set(cars.attrs["skipped"]["reason"])
    assert {"no_day0", "past_data", "bad_price", "no_benchmark"} <= reasons
    assert cars["car"].isna().any()


def test_matches_oracle_on_shuffled_rows_and_empty_inputs(prices, benchmarks, events):
    shuffled = prices.sample(frac=1, random_state=1)
    assert_same(shuffled, benchmarks, eligible(events).iloc[::-1], 5)
    assert_same(prices, benchmarks, events.iloc[:0], 5)
    assert_same(prices.iloc[:0], benchmarks, events.iloc[:50], 5)


@pytest.mark.slow
def test_matches_oracle_on_large_fake_universe(large_fake_prices, large_fake_events):
    benchmarks = make_synthetic_benchmarks("2015-01-01", "2026-12-31", seed=0)
    assert_same(large_fake_prices, benchmarks, eligible(large_fake_events), 5)
