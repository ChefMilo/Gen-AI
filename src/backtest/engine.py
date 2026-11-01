"""Deterministic event studies using the reference CAR definition."""

import numpy as np
import pandas as pd
from scipy import stats

from backtest.result import BacktestResult
from backtest.selection import select_events
from dsl.signal import SignalSpec
from evaluation.reference_car import event_cars


def run_event_study(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    events: pd.DataFrame,
    spec: SignalSpec,
    signal_id: str,
    min_events: int = 20,
) -> BacktestResult:
    """Measure a signal; session guardrails belong to the harness."""
    if min_events < 2:
        raise ValueError("min_events must be at least 2")

    selected = select_events(events, spec)

    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date price rows")
    if benchmarks.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date benchmark rows")

    # The reference calculator expects the benchmark ticker to be SPY.
    # Filter the requested benchmark first, then relabel a COPY.
    benchmark = benchmarks.loc[benchmarks["ticker"].eq(spec.benchmark)].copy()
    benchmark["ticker"] = "SPY"

    cars = event_cars(
        prices,
        benchmark,
        selected,
        spec.holding_days,
    )
    skipped = cars.attrs["skipped"].to_dict(orient="records")

    benchmark_dates = pd.DatetimeIndex(pd.to_datetime(benchmark["date"])).normalize().sort_values()

    ticker_dates = {
        ticker: pd.DatetimeIndex(pd.to_datetime(rows["date"])).normalize().sort_values()
        for ticker, rows in prices.groupby("ticker")
    }

    usable = []

    for row in cars.to_dict(orient="records"):
        reason = None
        day = pd.Timestamp(row["first_tradeable_date"]).normalize()

        # Do not silently stretch a five-day window across missing
        # stock observations when the benchmark traded that day.
        stock_days = ticker_dates[row["ticker"]]
        stock_start = stock_days.searchsorted(day)
        benchmark_start = benchmark_dates.searchsorted(day)

        actual = stock_days[stock_start : stock_start + spec.holding_days]
        expected = benchmark_dates[benchmark_start : benchmark_start + spec.holding_days]

        if not actual.equals(expected):
            reason = "missing_trading_day"
        elif not np.isfinite(row["car"]):
            reason = "nonfinite_return"

        if reason is not None:
            skipped.append(
                {
                    "event_id": row["event_id"],
                    "ticker": row["ticker"],
                    "first_tradeable_date": row["first_tradeable_date"],
                    "reason": reason,
                }
            )
        else:
            usable.append(row)

    result = BacktestResult(
        signal_id=signal_id,
        signal_spec=spec.model_copy(deep=True),
        status="insufficient_events",
        n_selected=len(selected),
        n_events=len(usable),
        event_ids=[row["event_id"] for row in usable],
        event_returns=usable,
        skipped_events=skipped,
    )

    if not usable:
        return result

    returns = np.array([row["car"] for row in usable])
    sign = 1 if spec.direction == "long" else -1

    result.mean_car = float(returns.mean())
    result.directional_mean_car = sign * result.mean_car
    result.hit_rate = float((sign * returns > 0).mean())

    if len(usable) < min_events:
        return result

    if np.unique(returns).size == 1:
        result.status = "zero_variance"
        return result

    test = stats.ttest_1samp(returns, popmean=0.0)
    t_stat = float(test.statistic)
    p_value = float(test.pvalue)

    if not np.isfinite([t_stat, p_value]).all():
        result.status = "invalid_statistics"
        return result

    result.status = "tested"
    result.t_stat = t_stat
    result.p_value_raw = p_value
    return result
