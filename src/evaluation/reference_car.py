"""Reference event study for Evaluation, until Member 4's backtest engine exists.

PROPOSED to Member 4 for the real engine. Definition:

- Abnormal return: AR_t = ret_t(stock) - ret_t(SPY), using the `ret` column only
  (daily return of adj_close) for both the stock and the benchmark.
- Window: starts on first_tradeable_date (day 0) and covers `holding_days` trading
  days for that ticker, i.e. the ticker's rows for days 0 .. holding_days-1.
- CAR: the sum of AR over the window (additive, not compounded).

An event is skipped, and listed in `cars.attrs["skipped"]` with a reason, when:
- "no_day0": the ticker has no prices row on first_tradeable_date;
- "past_data": the window runs past the ticker's last prices row;
- "bad_price": a window day has is_bad_price=True;
- "no_benchmark": SPY has no return on a window day.
"""

import math

import numpy as np
import pandas as pd

from store.schema import DEFAULT_BENCHMARK

CAR_COLUMNS = ["event_id", "ticker", "first_tradeable_date", "car", "n_days"]
SKIPPED_COLUMNS = ["event_id", "ticker", "first_tradeable_date", "reason"]


# Days per ticker in the (ticker, day) search key: more than any day number used.
_DAY_SPAN = 1 << 32


def _as_timestamps(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values)).dt.normalize()


def _day_numbers(timestamps: pd.Series) -> np.ndarray:
    """Days since 1970-01-01 (int64) of normalised timestamps."""
    return timestamps.to_numpy(dtype="datetime64[D]").astype(np.int64)


def event_cars(
    prices: pd.DataFrame, benchmarks: pd.DataFrame, events: pd.DataFrame, holding_days: int
) -> pd.DataFrame:
    """One row per measurable event, CAR_COLUMNS; skipped events in .attrs["skipped"].

    Vectorised: prices are sorted by (ticker, date) and every event's day 0 is found
    with one searchsorted on a (ticker, day) key; window sums gather positions.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")
    bench = benchmarks[benchmarks["ticker"] == DEFAULT_BENCHMARK]
    bench_ret = pd.Series(bench["ret"].to_numpy(), index=_as_timestamps(bench["date"]).to_numpy())

    tickers = pd.Index(pd.unique(prices["ticker"]))
    price_codes = tickers.get_indexer(prices["ticker"])
    price_dates = _as_timestamps(prices["date"])
    price_days = _day_numbers(price_dates)
    order = np.lexsort((price_days, price_codes))
    codes, days = price_codes[order], price_days[order]
    ret = prices["ret"].to_numpy(dtype=float)[order]
    bad = prices["is_bad_price"].to_numpy(dtype=bool)[order]
    spy = bench_ret.reindex(price_dates.to_numpy()[order]).to_numpy(dtype=float)
    n = len(order)
    keys = codes.astype(np.int64) * _DAY_SPAN + days
    group_end = np.searchsorted(codes, np.arange(len(tickers)), side="right")

    event_codes = tickers.get_indexer(events["ticker"])
    event_keys = event_codes.astype(np.int64) * _DAY_SPAN + _day_numbers(
        _as_timestamps(events["first_tradeable_date"])
    )
    known = event_codes >= 0
    pos = np.searchsorted(keys, event_keys)
    found = known & (pos < n)
    found[found] = keys[pos[found]] == event_keys[found]
    ticker_end = np.zeros(len(events), dtype=int)
    ticker_end[known] = group_end[event_codes[known]]
    past = found & (pos + holding_days > ticker_end)
    window = np.minimum(pos[:, None] + np.arange(holding_days), max(n - 1, 0))
    in_data = found & ~past
    bad_window = np.zeros(len(events), dtype=bool)
    no_bench = np.zeros(len(events), dtype=bool)
    car = np.full(len(events), np.nan)
    if in_data.any():
        rows = window[in_data]
        bad_window[in_data] = bad[rows].any(axis=1)
        no_bench[in_data] = np.isnan(spy[rows]).any(axis=1)
        car[in_data] = np.sum(ret[rows] - spy[rows], axis=1)

    reason = np.select(
        [~found, past, bad_window, no_bench],
        ["no_day0", "past_data", "bad_price", "no_benchmark"],
        default="",
    )
    key = {
        "event_id": list(events["event_id"]),
        "ticker": list(events["ticker"]),
        "first_tradeable_date": list(events["first_tradeable_date"]),
    }
    ok = reason == ""
    cars = [
        {"event_id": key["event_id"][i], "ticker": key["ticker"][i],
         "first_tradeable_date": key["first_tradeable_date"][i], "car": float(car[i]),
         "n_days": holding_days}
        for i in np.flatnonzero(ok)
    ]
    skipped = [
        {"event_id": key["event_id"][i], "ticker": key["ticker"][i],
         "first_tradeable_date": key["first_tradeable_date"][i], "reason": str(reason[i])}
        for i in np.flatnonzero(~ok)
    ]
    out = pd.DataFrame(cars, columns=CAR_COLUMNS)
    out.attrs["skipped"] = pd.DataFrame(skipped, columns=SKIPPED_COLUMNS)
    return out


def summary(cars: pd.DataFrame) -> dict:
    """n, mean CAR, t-stat of the mean (sample sd) and hit rate (share of CAR > 0)."""
    car = cars["car"].astype(float)
    n = len(car)
    if n == 0:
        return {"n": 0, "mean_car": None, "t_stat": None, "hit_rate": None}
    mean = float(car.mean())
    sd = float(car.std(ddof=1)) if n > 1 else float("nan")
    t_stat = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None
    return {"n": n, "mean_car": mean, "t_stat": t_stat, "hit_rate": float((car > 0).mean())}
