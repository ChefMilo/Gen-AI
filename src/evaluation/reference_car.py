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


def _as_timestamps(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values)).dt.normalize()


def event_cars(
    prices: pd.DataFrame, benchmarks: pd.DataFrame, events: pd.DataFrame, holding_days: int
) -> pd.DataFrame:
    """One row per measurable event, CAR_COLUMNS; skipped events in .attrs["skipped"]."""
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
