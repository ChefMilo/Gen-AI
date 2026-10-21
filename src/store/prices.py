"""Price and calendar helpers (docs/data_schema.md, "Helpers").

Use these instead of reading the parquet files directly.
"""

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from store.schema import DEFAULT_BENCHMARK, Benchmark

DATA_DIR_ENV = "GENAI_DATA_DIR"


def data_dir() -> Path:
    """Local path of the shared Drive folder "Gen AI/Data/", from $GENAI_DATA_DIR."""
    value = os.environ.get(DATA_DIR_ENV)
    if not value:
        raise RuntimeError(
            f"Set {DATA_DIR_ENV} to your local copy of the shared Drive folder 'Gen AI/Data/'."
        )
    return Path(value)


# TODO(member-1): copy the three helpers below from notebook 03, section 10.


def load_prices(
    tickers: list[str] | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    drop_bad: bool = True,
    usable_only: bool = True,
) -> pd.DataFrame:
    """prices/prices.parquet without bad price days and dropped companies."""
    raise NotImplementedError


def load_benchmarks(ticker: Benchmark = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """prices/benchmarks.parquet for one of SPY (default), IWM, MDY."""
    raise NotImplementedError


def first_tradeable_date(acceptance_et: datetime) -> date:
    """Same day if accepted before 16:00 ET on a trading day, else the next trading day."""
    raise NotImplementedError
