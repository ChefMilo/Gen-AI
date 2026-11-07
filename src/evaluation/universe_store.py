"""Build and cache an experiment's universes under <root>/universes/<universe_id>/.

Each universe directory holds:
- record.json: the UniverseRecord (truths, dropped targets, placebo info);
- prices_delta.parquet (planted only): the prices rows whose ret was planted, as
  (row, ret), with row the position in the base prices. Planting also rebuilds
  adj_close from the first planted row of each ticker, which changes nearly every
  later row, so adj_close is not stored: load_universe replays make_planted's rebuild,
  per spec in planting order and per ticker from that spec's first window row, on the
  final ret. That gives the same floats, because every row's final adj_close comes
  from the last rebuild covering it, over rows no later spec changes;
- nothing for an entry's noise_seed: its prices start from
  universes.replicate.resample_idiosyncratic(base prices, seed), redrawn on load;
- events_delta.parquet (placebo only): the moved events' changed columns
  (PLACEBO_CHANGED), by event_id. Dropped events are listed in the record;
- meta.json, written last: the cache key (plan entry + data fingerprint + FORMAT_VERSION),
  hashes of the universe's prices and events, build time and file sizes.

ensure_universe reuses a directory whose meta has the same cache key and rebuilds it
otherwise. load_universe rebuilds the full frames from the base data plus the delta
and checks them against the hashes in meta, so a cache that does not reproduce the
universe exactly raises instead of being used.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from evaluation.plan import DataFingerprint, UniverseEntry, frame_hash, prices_hash
from evaluation.universe_record import (
    UniverseRecord,
    placebo_record,
    plant_universe,
    real_record,
)
from universes.placebo import DATE_FIELDS, make_placebo
from universes.planted import window_positions
from universes.replicate import resample_idiosyncratic

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
UNIVERSES_DIR = "universes"
RECORD, META = "record.json", "meta.json"
PRICES_DELTA, EVENTS_DELTA = "prices_delta.parquet", "events_delta.parquet"
CACHE_FILES = [RECORD, META, PRICES_DELTA, EVENTS_DELTA]
PLACEBO_CHANGED = ["first_tradeable_date", "acceptance_datetime", "mcap_usd", "mcap_bucket",
                   "buyback_pct_mcap", *DATE_FIELDS]
EVENT_ORDER = ["first_tradeable_date", "ticker", "event_id"]


@dataclass(frozen=True)
class BaseData:
    """The unplanted prices, benchmarks and events every universe is built from.

    real (optional): other data for the real universe only (data.real: store in the
    config), with its fingerprint. The planted and placebo universes ignore it.
    """

    prices: pd.DataFrame
    benchmarks: pd.DataFrame
    events: pd.DataFrame
    real: "BaseData | None" = None
    real_fingerprint: DataFingerprint | None = None


@dataclass
class Universe:
    entry: UniverseEntry
    record: UniverseRecord
    prices: pd.DataFrame
    benchmarks: pd.DataFrame
    events: pd.DataFrame


class BuildMeta(BaseModel):
    format_version: int
    cache_key: str
    entry: UniverseEntry
    fingerprint: DataFingerprint
    prices_hash: str  # of the universe's prices
    events_hash: str  # of the universe's events
    build_seconds: float
    file_bytes: dict[str, int]


@dataclass(frozen=True)
class BuildStatus:
    universe_id: str
    universe_type: str
    status: Literal["built", "reused"]
    seconds: float
    bytes: int  # on disk, all files of the universe


class StaleCacheError(RuntimeError):
    """A cached universe does not belong to this entry and data, or does not reproduce."""


def ensure_git_ignored(path: Path) -> None:
    """Raise if path lies in a git work tree and git would not ignore it.

    Paths outside any repository (e.g. temp dirs) pass. So does a missing git, with a
    warning.
    """
    path = Path(path).resolve()
    cwd = next(p for p in [path, *path.parents] if p.exists())
    try:
        result = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=cwd,
                                capture_output=True, text=True)
    except FileNotFoundError:
        log.warning("git not found; cannot check that %s is git-ignored", path)
        return
    if result.returncode == 1:
        raise RuntimeError(
            f"{path} is inside a git work tree and not git-ignored; add it to .gitignore "
            "(results/ is ignored from 'Ignore generated results/ output') before writing"
        )


def universe_dir(root: Path, universe_id: str) -> Path:
    return Path(root) / UNIVERSES_DIR / universe_id


def entry_data(entry: UniverseEntry, data: BaseData,
               fingerprint: DataFingerprint) -> tuple[BaseData, DataFingerprint]:
    """The data and fingerprint entry is built from: data.real for the real universe when
    it is set, so its cache follows the real data; the base data otherwise."""
    if entry.universe_type == "real" and data.real is not None:
        return data.real, data.real_fingerprint
    return data, fingerprint


def entry_prices(entry: UniverseEntry, data: BaseData) -> pd.DataFrame:
    """The prices entry starts from: the base prices, or (with a noise_seed) their
    independent redraw (universes.replicate). Deterministic, so never stored."""
    if entry.noise_seed is None:
        return data.prices
    return resample_idiosyncratic(data.prices, data.benchmarks, entry.noise_seed)


def build_universe(entry: UniverseEntry, data: BaseData,
                   base_prices: pd.DataFrame | None = None) -> Universe:
    """The universe for entry, built from the base data (nothing is read or written)."""
    base = entry_prices(entry, data) if base_prices is None else base_prices
    prices, events = base, data.events
    if entry.universe_type == "planted":
        prices, record = plant_universe(
            entry.universe_id, base, data.events, entry.specs, entry.min_events,
            seed=entry.seed, benchmarks=data.benchmarks,
        )
    elif entry.universe_type == "placebo":
        events, info = make_placebo(base, data.events, entry.seed, entry.min_gap_days,
                                    entry.holding_days)
        record = placebo_record(entry.universe_id, info)
    else:
        record = real_record(entry.universe_id)
    return Universe(entry, record, prices, data.benchmarks, events)


def _changed(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return ~((before == after) | (np.isnan(before) & np.isnan(after)))


def prices_delta(base: pd.DataFrame, planted: pd.DataFrame) -> pd.DataFrame:
    """(row, ret) for the rows whose ret differs from the base prices."""
    if not base.index.equals(planted.index):
        raise ValueError("planted prices must keep the base rows in the same order")
    before = base["ret"].to_numpy(dtype=float)
    after = planted["ret"].to_numpy(dtype=float)
    rows = np.flatnonzero(_changed(before, after))
    return pd.DataFrame({"row": rows.astype(np.int64), "ret": after[rows]})


def apply_prices_delta(
    base: pd.DataFrame, events: pd.DataFrame, record: UniverseRecord, delta: pd.DataFrame
) -> pd.DataFrame:
    """The planted prices: base with delta's ret, and adj_close rebuilt as make_planted does."""
    out = base.copy()
    ret = out["ret"].to_numpy(dtype=float, copy=True)
    adj_close = out["adj_close"].to_numpy(dtype=float, copy=True)
    ret[delta["row"].to_numpy(dtype=np.int64)] = delta["ret"].to_numpy(dtype=float)

    dates = pd.to_datetime(base["date"]).dt.normalize().to_numpy()
    ticker_rows = {
        ticker: rows[np.argsort(dates[rows], kind="stable")]
        for ticker, rows in base.groupby("ticker", sort=False).indices.items()
    }
    position = np.empty(len(base), dtype=np.int64)  # row -> its place in its ticker's rows
    for rows in ticker_rows.values():
        position[rows] = np.arange(len(rows))
    tickers = base["ticker"].to_numpy()

    for truth in record.truths:
        targets = events[events["event_id"].isin(set(truth.target_event_ids))]
        first: dict[str, int] = {}
        for window in window_positions(base, targets, truth.spec.holding_days).values():
            ticker = tickers[window[0]]
            first[ticker] = min(first.get(ticker, len(base)), int(position[window[0]]))
        for ticker, start in first.items():
            rows = ticker_rows[ticker]
            start = max(start, 1)
            adj_close[rows[start:]] = adj_close[rows[start - 1]] * np.cumprod(1 + ret[rows[start:]])
    out["ret"] = ret
    out["adj_close"] = adj_close
    return out


def events_delta(placebo: pd.DataFrame, record: UniverseRecord) -> pd.DataFrame:
    """The moved events' PLACEBO_CHANGED columns, by event_id."""
    moved = list(record.placebo_info.original_first_tradeable_date)
    rows = placebo[placebo["event_id"].isin(set(moved))]
    return rows[["event_id", *PLACEBO_CHANGED]].reset_index(drop=True)


def apply_events_delta(
    base: pd.DataFrame, record: UniverseRecord, delta: pd.DataFrame
) -> pd.DataFrame:
    """The placebo events: base with the placebo columns, moved rows and drops applied."""
    out = base.reset_index(drop=True).copy()
    out["placebo"] = True
    out["original_first_tradeable_date"] = out["first_tradeable_date"]
    where = pd.Series(out.index, index=out["event_id"])
    pos = where.loc[delta["event_id"]].to_numpy()
    for column in PLACEBO_CHANGED:
        values = delta[column].to_numpy(dtype=object)
        for p, value in zip(pos, values, strict=True):
            out.at[p, column] = None if _missing(value) else value
    dropped = {d.event_id for d in record.placebo_info.dropped}
    out = out[~out["event_id"].isin(dropped)]
    return out.sort_values(EVENT_ORDER, ignore_index=True)


def _missing(value) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_meta(path: Path) -> BuildMeta | None:
    meta = path / META
    if not meta.exists():
        return None
    try:
        return BuildMeta.model_validate_json(meta.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _expected_files(entry: UniverseEntry) -> list[str]:
    extra = {"planted": [PRICES_DELTA], "placebo": [EVENTS_DELTA]}.get(entry.universe_type, [])
    return [RECORD, *extra]


def is_cached(root: Path, entry: UniverseEntry, fingerprint: DataFingerprint) -> bool:
    path = universe_dir(root, entry.universe_id)
    meta = read_meta(path)
    return (meta is not None and meta.cache_key == entry.cache_key(fingerprint, FORMAT_VERSION)
            and all((path / f).exists() for f in _expected_files(entry)))


def _dir_bytes(path: Path) -> int:
    return sum((path / f).stat().st_size for f in CACHE_FILES if (path / f).exists())


def ensure_universe(
    root: Path, entry: UniverseEntry, fingerprint: DataFingerprint, data: BaseData
) -> BuildStatus:
    """Reuse the cached universe if its key matches; build and write it otherwise."""
    data, fingerprint = entry_data(entry, data, fingerprint)
    path = universe_dir(root, entry.universe_id)
    if is_cached(root, entry, fingerprint):
        return BuildStatus(entry.universe_id, entry.universe_type, "reused", 0.0,
                           _dir_bytes(path))

    start = time.perf_counter()
    path.mkdir(parents=True, exist_ok=True)
    (path / META).unlink(missing_ok=True)  # first, so a half-written cache is never reused
    for name in CACHE_FILES:
        (path / name).unlink(missing_ok=True)
    base = entry_prices(entry, data)
    universe = build_universe(entry, data, base)
    _write_text(universe.record.model_dump_json(), path / RECORD)
    if entry.universe_type == "planted":
        _write_parquet(prices_delta(base, universe.prices), path / PRICES_DELTA)
    elif entry.universe_type == "placebo":
        _write_parquet(events_delta(universe.events, universe.record), path / EVENTS_DELTA)
    seconds = time.perf_counter() - start
    meta = BuildMeta(
        format_version=FORMAT_VERSION,
        cache_key=entry.cache_key(fingerprint, FORMAT_VERSION),
        entry=entry,
        fingerprint=fingerprint,
        prices_hash=prices_hash(universe.prices),
        events_hash=frame_hash(universe.events),
        build_seconds=seconds,
        file_bytes={f: (path / f).stat().st_size for f in _expected_files(entry)},
    )
    _write_text(meta.model_dump_json(indent=1), path / META)
    return BuildStatus(entry.universe_id, entry.universe_type, "built", seconds,
                       _dir_bytes(path))


def load_universe(
    root: Path, entry: UniverseEntry, fingerprint: DataFingerprint, data: BaseData,
    verify: bool = True,
) -> Universe:
    """The cached universe, rebuilt from the base data plus its delta.

    Raises StaleCacheError if the cache belongs to another entry or other data, or
    (with verify) if the rebuilt frames do not match the hashes taken at build time.
    """
    data, fingerprint = entry_data(entry, data, fingerprint)
    path = universe_dir(root, entry.universe_id)
    if not is_cached(root, entry, fingerprint):
        raise StaleCacheError(f"{entry.universe_id}: no cache for this plan entry and data")
    meta = read_meta(path)
    record = UniverseRecord.model_validate_json((path / RECORD).read_text(encoding="utf-8"))
    prices, events = entry_prices(entry, data), data.events
    if entry.universe_type == "planted":
        prices = apply_prices_delta(prices, data.events, record,
                                    pd.read_parquet(path / PRICES_DELTA))
    elif entry.universe_type == "placebo":
        events = apply_events_delta(data.events, record, pd.read_parquet(path / EVENTS_DELTA))
    if verify:
        if prices_hash(prices) != meta.prices_hash:
            raise StaleCacheError(f"{entry.universe_id}: rebuilt prices differ from the build")
        if frame_hash(events) != meta.events_hash:
            raise StaleCacheError(f"{entry.universe_id}: rebuilt events differ from the build")
    return Universe(entry, record, prices, data.benchmarks, events)


def build_universes(
    root: Path, entries: list[UniverseEntry], fingerprint: DataFingerprint, data: BaseData
) -> list[BuildStatus]:
    statuses = []
    for entry in entries:
        status = ensure_universe(root, entry, fingerprint, data)
        log.info("%s %s (%s): %.1fs, %.0f KB", status.status, entry.universe_id,
                 entry.universe_type, status.seconds, status.bytes / 1024)
        statuses.append(status)
    return statuses


def write_json(obj, path: Path) -> None:
    """obj (a Pydantic model or JSON-able data) to path, atomically."""
    text = obj.model_dump_json(indent=1) if isinstance(obj, BaseModel) else json.dumps(
        obj, indent=1, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(text, path)
