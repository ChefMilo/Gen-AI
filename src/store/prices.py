"""Price and calendar helpers (docs/data_schema.md, "Helpers").

Use these instead of reading the parquet files directly. They mirror notebook 03, section 10.
"""

import bisect
import os
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import get_args

import pandas as pd

from store.schema import DEFAULT_BENCHMARK, Benchmark

DATA_DIR_ENV = "GENAI_DATA_DIR"

PRICES = Path("prices/prices.parquet")
BENCHMARKS = Path("prices/benchmarks.parquet")
TRADING_CALENDAR = Path("prices/trading_calendar.csv")
UNIVERSE_QUALITY = Path("universe_quality.csv")
USABLE = ("keep", "keep_from")  # universe_quality.csv recommendations that are used

MARKET_CLOSE = time(16, 0)  # US Eastern
EASTERN = "America/New_York"

DateLike = date | datetime | str


def data_dir() -> Path:
    """Local path of the shared Drive folder "Gen AI/Data/", from $GENAI_DATA_DIR."""
    value = os.environ.get(DATA_DIR_ENV)
    if not value:
        raise RuntimeError(
            f"Set {DATA_DIR_ENV} to your local copy of the shared Drive folder 'Gen AI/Data/'."
        )
    return Path(value)


def _between(df: pd.DataFrame, start: DateLike | None, end: DateLike | None) -> pd.DataFrame:
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df


def _usable_rows(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    """Rows of keep / keep_from companies on or after their usable_from date."""
    columns = ["ticker", "recommendation", "usable_from"]
    quality = pd.read_csv(base / UNIVERSE_QUALITY, usecols=columns)
    quality = quality[quality["recommendation"].isin(USABLE)].set_index("ticker")
    usable_from = df["ticker"].map(pd.to_datetime(quality["usable_from"]))
    keep = df["ticker"].isin(quality.index) & (usable_from.isna() | (df["date"] >= usable_from))
    return df[keep]


def load_prices(
    tickers: Iterable[str] | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    drop_bad: bool = True,
    usable_only: bool = True,
) -> pd.DataFrame:
    """prices/prices.parquet without bad price days and dropped companies.

    drop_bad: drop is_bad_price days (new listings, >100% daily moves, broken histories).
    usable_only: keep only companies marked keep / keep_from in universe_quality.csv, from
        their usable_from date. usable_from is 2016-01-01 for most companies, so pass
        usable_only=False for 2015 look-back windows (bad days are still dropped by drop_bad).
    start, end: inclusive date range.
    """
    base = data_dir()
    df = pd.read_parquet(base / PRICES)
    df["date"] = pd.to_datetime(df["date"])
    if tickers is not None:
        df = df[df["ticker"].isin(list(tickers))]
    if drop_bad:
        df = df[~df["is_bad_price"].fillna(False).astype(bool)]
    if usable_only:
        df = _usable_rows(df, base)
    df = _between(df, start, end)
    return df.sort_values(["ticker", "date"], ignore_index=True)


def load_benchmarks(ticker: Benchmark = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """prices/benchmarks.parquet for one of SPY (default), IWM, MDY."""
    if ticker not in get_args(Benchmark):
        raise ValueError(f"unknown benchmark {ticker!r}; use one of {get_args(Benchmark)}")
    df = pd.read_parquet(data_dir() / BENCHMARKS)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"] == ticker]
    return df.sort_values("date", ignore_index=True)


def trading_calendar() -> list[date]:
    """NYSE trading days from prices/trading_calendar.csv, sorted."""
    return list(_read_calendar(str(data_dir())))


@lru_cache(maxsize=4)
def _read_calendar(base: str) -> tuple[date, ...]:
    days = pd.read_csv(Path(base) / TRADING_CALENDAR, parse_dates=["date"])["date"]
    return tuple(sorted(d.date() for d in days))


def first_tradeable_date(acceptance_et: datetime, calendar: Sequence[date] | None = None) -> date:
    """Same day if accepted before 16:00 ET on a trading day, else the next trading day.

    acceptance_et: SEC acceptance time in US Eastern (naive); a timezone-aware time is
        converted to US Eastern first.
    calendar: sorted trading days; defaults to prices/trading_calendar.csv.
    """
    ts = pd.Timestamp(acceptance_et)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(EASTERN).tz_localize(None)
    days = _read_calendar(str(data_dir())) if calendar is None else calendar
    day = ts.date()
    i = bisect.bisect_left(days, day)
    if i < len(days) and days[i] == day and ts.time() < MARKET_CLOSE:
        return day
    i = bisect.bisect_right(days, day)
    if i >= len(days):
        raise ValueError(f"no trading day after {acceptance_et} in the calendar")
    return days[i]
