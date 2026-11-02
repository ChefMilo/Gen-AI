"""The event store: events/events.parquet in the shared Drive folder (docs/data_schema.md).

load_events() returns the same columns as universes.fake_events.make_fake_events, so the
runner can switch from fake to real events.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from store.prices import UNIVERSE_QUALITY, USABLE, data_dir
from store.schema import Event

EVENTS = Path("events/events.parquet")
JSON_COLUMNS = ("text_features", "evidence")  # dicts, stored as JSON text
DATE_COLUMNS = (
    "first_tradeable_date",
    "announcement_date",
    "record_date",
    "payable_date",
    "ex_date",
    "buyback_expiry",
)


def save_events(events: pd.DataFrame, path: Path | None = None) -> Path:
    """Write events (Event columns) to events/events.parquet, or path."""
    path = Path(path) if path else data_dir() / EVENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    out = events.reindex(columns=list(Event.model_fields)).copy()
    for column in JSON_COLUMNS:
        out[column] = out[column].map(lambda d: json.dumps(d if isinstance(d, dict) else {}))
    out["flags"] = out["flags"].map(lambda f: list(f) if f is not None else [])
    for column in DATE_COLUMNS:
        out[column] = pd.to_datetime(out[column])
    out.to_parquet(path, index=False)
    return path


def _usable_rows(events: pd.DataFrame, base: Path) -> pd.DataFrame:
    quality = pd.read_csv(
        base / UNIVERSE_QUALITY, usecols=["ticker", "recommendation", "usable_from"]
    )
    quality = quality[quality["recommendation"].isin(USABLE)].set_index("ticker")
    usable_from = events["ticker"].map(pd.to_datetime(quality["usable_from"]))
    day = pd.to_datetime(events["first_tradeable_date"])
    keep = events["ticker"].isin(quality.index) & (usable_from.isna() | (day >= usable_from))
    return events[keep]


def load_events(
    start: date | str | None = None,
    end: date | str | None = None,
    usable_only: bool = True,
    path: Path | None = None,
) -> pd.DataFrame:
    """events/events.parquet as a DataFrame with the Event columns.

    usable_only: only what backtests should use: is_new_announcement, extraction_status
        "passed", and companies marked keep / keep_from in universe_quality.csv from
        their usable_from date.
    start, end: inclusive range of first_tradeable_date.
    """
    base = data_dir() if path is None else None
    events = pd.read_parquet(path or base / EVENTS)
    for column in JSON_COLUMNS:
        events[column] = events[column].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    events["flags"] = events["flags"].map(lambda f: list(f) if f is not None else [])
    for column in DATE_COLUMNS:
        events[column] = pd.to_datetime(events[column]).dt.date.astype(object)
        events[column] = events[column].where(events[column].notna(), None)
    if usable_only:
        events = events[
            events["is_new_announcement"].astype(bool) & (events["extraction_status"] == "passed")
        ]
        events = _usable_rows(events, base if base is not None else data_dir())
    day = pd.to_datetime(events["first_tradeable_date"])
    if start is not None:
        events = events[day >= pd.Timestamp(start)]
        day = day[events.index]
    if end is not None:
        events = events[day <= pd.Timestamp(end)]
    events = events.reindex(columns=list(Event.model_fields))
    return events.sort_values(["first_tradeable_date", "ticker", "event_id"], ignore_index=True)
