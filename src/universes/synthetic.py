"""Synthetic prices and benchmarks in the prices/ parquet layout (docs/data_schema.md).

For tests and for building against fake events before the real data is wired in.
All randomness comes from numpy Generators seeded from `seed`, so the same
arguments always give the same frames.
"""

from datetime import date

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "ret",
    "volume",
    "raw_close",
    "split_factor",
    "shares_outstanding",
    "shares_source",
    "mcap_usd",
    "mcap_bucket",
    "is_bad_price",
    "bad_price_reason",
]
BENCHMARK_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "ret"]
BENCHMARKS = ("SPY", "IWM", "MDY")
BAD_PRICE_REASONS = ("new_listing", "extreme_return", "known_broken_history")

# Seed streams, so prices and benchmarks share one market factor.
_MARKET_STREAM = 0
_BENCHMARK_STREAM = 1
_TICKER_STREAM = 2


class _NyseLikeCalendar(AbstractHolidayCalendar):
    """NYSE full-day holidays, close enough for synthetic data."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def trading_days(start: date | str, end: date | str) -> pd.DatetimeIndex:
    """NYSE-like trading days from start to end, inclusive."""
    holidays = _NyseLikeCalendar().holidays(start, end)
    return pd.bdate_range(start, end, freq="C", holidays=holidays)


def _market_returns(n_days: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, _MARKET_STREAM])
    return rng.normal(0.0003, 0.01, n_days)


def _dividend_factor(n_days: int, rng: np.random.Generator, yield_range: tuple[float, float]):
    """close / adj_close: grows by (1 + quarterly yield) for each ex-date after a day."""
    factor = np.ones(n_days)
    first_ex = int(rng.integers(1, 63))
    for ex in range(first_ex, n_days, 63):
        factor[:ex] *= 1 + rng.uniform(*yield_range)
    return factor


def _ohlc(close: np.ndarray, rng: np.random.Generator):
    prev_close = np.concatenate([[close[0]], close[:-1]])
    open_ = prev_close * (1 + rng.normal(0, 0.004, len(close)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(close))))
    return open_, high, low


def _ret_from_adj_close(adj_close: np.ndarray) -> np.ndarray:
    ret = np.empty_like(adj_close)
    ret[0] = np.nan
    ret[1:] = adj_close[1:] / adj_close[:-1] - 1
    return ret


def make_synthetic_prices(
    tickers: list[str], start: date | str, end: date | str, seed: int
) -> pd.DataFrame:
    """Fake prices/prices.parquet: one row per ticker per trading day, PRICE_COLUMNS."""
    days = trading_days(start, end)
    n = len(days)
    market = _market_returns(n, seed)
    frames = []
    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng([seed, _TICKER_STREAM, i])

        beta = rng.uniform(0.6, 1.4)
        ret = beta * market + rng.normal(0, rng.uniform(0.012, 0.025), n)
        adj_close = rng.uniform(20, 150) * np.cumprod(1 + ret)
        close = adj_close * _dividend_factor(n, rng, (0.003, 0.01))
        open_, high, low = _ohlc(close, rng)

        # About one ticker in ten has a 2:1 or 3:2 split; close is split-adjusted.
        split_factor = np.ones(n)
        if rng.random() < 0.1 and n > 2:
            split_factor[: int(rng.integers(1, n))] = rng.choice([2.0, 1.5])
        base_shares = np.exp(rng.uniform(np.log(5e6), np.log(2e8)))
        shares = base_shares / split_factor
        raw_close = close * split_factor

        is_bad = np.zeros(n, dtype=bool)
        reason = np.full(n, None, dtype=object)
        if rng.random() < 0.1:
            is_bad[:5] = True
            reason[:5] = "new_listing"
        for day in rng.choice(n, size=min(n, int(rng.poisson(2))), replace=False):
            if not is_bad[day]:
                is_bad[day] = True
                reason[day] = rng.choice(["extreme_return", "known_broken_history"])

        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": adj_close,
                    "ret": _ret_from_adj_close(adj_close),
                    "volume": rng.lognormal(np.log(3e5), 0.5, n).astype("int64"),
                    "raw_close": raw_close,
                    "split_factor": split_factor,
                    "shares_outstanding": shares,
                    "shares_source": "yahoo" if rng.random() < 0.08 else "sec",
                    "mcap_usd": raw_close * shares,
                    "is_bad_price": is_bad,
                    "bad_price_reason": reason,
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    prices = pd.concat(frames, ignore_index=True)
    pct = prices.groupby("date")["mcap_usd"].rank(method="first", pct=True)
    prices["mcap_bucket"] = np.select([pct <= 1 / 3, pct <= 2 / 3], ["small", "mid"], "large")
    return prices.sort_values(["date", "ticker"], ignore_index=True)[PRICE_COLUMNS]


def make_synthetic_benchmarks(start: date | str, end: date | str, seed: int) -> pd.DataFrame:
    """Fake prices/benchmarks.parquet for SPY, IWM and MDY, BENCHMARK_COLUMNS.

    Built on the same market factor as make_synthetic_prices for the same seed.
    """
    days = trading_days(start, end)
    n = len(days)
    market = _market_returns(n, seed)
    rng = np.random.default_rng([seed, _BENCHMARK_STREAM])
    specs = {"SPY": (1.0, 0.0, 200.0), "IWM": (1.15, 0.006, 110.0), "MDY": (1.05, 0.004, 260.0)}
    frames = []
    for ticker in BENCHMARKS:
        beta, noise, level = specs[ticker]
        ret = beta * market + rng.normal(0, noise, n)
        adj_close = level * np.cumprod(1 + ret)
        close = adj_close * _dividend_factor(n, rng, (0.002, 0.004))
        open_, high, low = _ohlc(close, rng)
        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": adj_close,
                    "volume": rng.lognormal(np.log(5e7), 0.4, n).astype("int64"),
                    "ret": _ret_from_adj_close(adj_close),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[BENCHMARK_COLUMNS]
