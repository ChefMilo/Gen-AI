from datetime import date

import numpy as np
import pandas as pd
import pytest

from universes.synthetic import (
    BAD_PRICE_REASONS,
    BENCHMARK_COLUMNS,
    PRICE_COLUMNS,
    make_synthetic_benchmarks,
    make_synthetic_prices,
    trading_days,
)

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"


@pytest.fixture(scope="module")
def prices():
    return make_synthetic_prices(TICKERS, START, END, seed=0)


@pytest.fixture(scope="module")
def benchmarks():
    return make_synthetic_benchmarks(START, END, seed=0)


def test_price_columns_match_data_schema(prices):
    assert list(prices.columns) == PRICE_COLUMNS
    assert prices["volume"].dtype == np.int64
    assert set(prices["shares_source"]) <= {"sec", "yahoo"}


def test_benchmark_columns_and_tickers(benchmarks):
    assert list(benchmarks.columns) == BENCHMARK_COLUMNS
    assert set(benchmarks["ticker"]) == {"SPY", "IWM", "MDY"}


def test_same_seed_same_frames(prices, benchmarks):
    pd.testing.assert_frame_equal(prices, make_synthetic_prices(TICKERS, START, END, seed=0))
    pd.testing.assert_frame_equal(benchmarks, make_synthetic_benchmarks(START, END, seed=0))
    other = make_synthetic_prices(TICKERS, START, END, seed=1)
    assert not np.allclose(prices["adj_close"], other["adj_close"])


def test_trading_days_skip_weekends_and_nyse_holidays():
    days = set(trading_days("2019-01-01", "2023-12-31").date)
    assert all(d.weekday() < 5 for d in days)
    for holiday in [
        date(2019, 1, 1),
        date(2019, 4, 19),  # Good Friday
        date(2019, 7, 4),
        date(2019, 11, 28),
        date(2019, 12, 25),
        date(2023, 6, 19),  # Juneteenth
    ]:
        assert holiday not in days
    assert date(2019, 7, 5) in days
    assert date(2021, 6, 18) in days  # before Juneteenth became a holiday


def test_every_ticker_trades_every_day(prices):
    n_days = len(trading_days(START, END))
    assert (prices.groupby("ticker").size() == n_days).all()


@pytest.mark.parametrize("frame", ["prices", "benchmarks"])
def test_ret_is_daily_return_of_adj_close(frame, request):
    df = request.getfixturevalue(frame).sort_values(["ticker", "date"])
    expected = df.groupby("ticker")["adj_close"].pct_change()
    np.testing.assert_allclose(df["ret"], expected, rtol=1e-12, equal_nan=True)
    assert df.groupby("ticker")["ret"].head(1).isna().all()


def test_ohlc_bounds(prices):
    assert (prices["high"] >= prices[["open", "close"]].max(axis=1)).all()
    assert (prices["low"] <= prices[["open", "close"]].min(axis=1)).all()
    assert (prices["low"] > 0).all()


def test_raw_close_and_mcap(prices):
    np.testing.assert_allclose(prices["raw_close"], prices["close"] * prices["split_factor"])
    np.testing.assert_allclose(
        prices["mcap_usd"], prices["raw_close"] * prices["shares_outstanding"]
    )
    assert (prices["split_factor"] >= 1).all()


def test_mcap_bucket_is_thirds_on_each_date(prices):
    for _, day in prices.groupby("date"):
        counts = day["mcap_bucket"].value_counts()
        assert set(counts.index) == {"small", "mid", "large"}
        assert counts.max() - counts.min() <= 1
        by_bucket = day.groupby("mcap_bucket")["mcap_usd"]
        assert by_bucket.max()["small"] < by_bucket.min()["mid"]
        assert by_bucket.max()["mid"] < by_bucket.min()["large"]


def test_a_few_bad_price_days(prices):
    bad = prices["is_bad_price"]
    assert 0 < bad.mean() < 0.01
    assert set(prices.loc[bad, "bad_price_reason"]) <= set(BAD_PRICE_REASONS)
    assert prices.loc[~bad, "bad_price_reason"].isna().all()
