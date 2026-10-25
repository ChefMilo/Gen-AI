"""Placebo universes: real events moved to random dates, so no event effect exists.

make_placebo keeps the prices and every extracted field of each event, and gives
each eligible event (is_new_announcement and passed) a new first_tradeable_date
drawn without replacement from its ticker's valid days. A valid day has a clean
prices row, a clean prior row (for the point-in-time fields) and a full window of
holding_days clean rows, and lies at least min_gap_days trading days from every
real event date of that ticker. With min_gap_days >= holding_days, no placebo
window shares a day with a real event's window, so an effect planted on the real
events (universes.planted) never reaches a placebo event. Every finding on a
placebo universe is false by construction.

acceptance_datetime moves with the event and keeps its clock time, so that
first_tradeable_date still follows from it (docs/data_schema.md). mcap_usd,
mcap_bucket and buyback_pct_mcap are recomputed from the prior trading day;
dividend_change_pct and the other extracted fields are left as they were.
Ineligible rows are kept unchanged and stay ineligible.

The other date fields (DATE_FIELDS) move with the event: by the same number of
trading days as first_tradeable_date moved on the ticker's rows, counted on the
NYSE calendar of the prices (shift_trading_days). A date that would leave the
calendar is shifted by calendar days instead; announcement_date is kept on or
before first_tradeable_date and payable_date on or after record_date. Both
adjustments are listed in PlaceboInfo.date_adjustments.

store.schema.Event ignores unknown fields, so the extra columns (placebo,
original_first_tradeable_date) are dropped by rows_as_events(); the original dates
are in PlaceboInfo as well.
"""

import bisect
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from store import prices as store_prices
from universes.fake_events import MARKET_CLOSE
from universes.fake_events import first_tradeable_date as local_first_tradeable_date
from universes.planted import DEFAULT_HOLDING_DAYS, eligible

DEFAULT_MIN_GAP_DAYS = 10
PLACEBO_COLUMNS = ["placebo", "original_first_tradeable_date"]
DATE_FIELDS = ["announcement_date", "record_date", "payable_date", "ex_date", "buyback_expiry"]

_PLACEBO_STREAM = 4


class DroppedEvent(BaseModel):
    event_id: str
    ticker: str
    n_valid_days: int  # valid days the ticker had, fewer than its eligible events


class DateAdjustment(BaseModel):
    """A date field not shifted by trading days alone."""

    event_id: str
    field: str
    original: date
    shifted: date
    # calendar_days: a trading-day shift would leave the prices calendar.
    # clamped: moved to keep announcement <= first_tradeable and record <= payable.
    method: Literal["calendar_days", "clamped"]


class PlaceboInfo(BaseModel):
    seed: int
    min_gap_days: int
    holding_days: int
    n_eligible: int  # eligible events in the input
    n_moved: int  # eligible events in the output, all on new dates
    original_first_tradeable_date: dict[str, date]  # moved event_id -> real date
    valid_days: dict[str, int]  # ticker -> valid days it had
    dropped: list[DroppedEvent]  # eligible events left out: their ticker ran out of days
    shift_trading_days: dict[str, int]  # moved event_id -> trading days it moved
    date_adjustments: list[DateAdjustment]


def shift_trading_days(day: date, k: int, calendar: list[date]) -> date | None:
    """day moved by k trading days; None if that leaves the calendar.

    A non-trading day keeps its calendar-day offset from the last trading day on or
    before it, cut short if the new gap to the next trading day is shorter, so the
    result always has the same trading-day position + k (it may land on a trading day).
    """
    if not calendar[0] <= day <= calendar[-1]:
        return None
    i = bisect.bisect_right(calendar, day) - 1
    j = i + k
    if not 0 <= j < len(calendar):
        return None
    offset = (day - calendar[i]).days
    if offset == 0:
        return calendar[j]
    if j == len(calendar) - 1:
        return None
    room = (calendar[j + 1] - calendar[j]).days - 1
    return calendar[j] + timedelta(days=min(offset, room))


def _shift_dates(
    event: pd.Series, new_day: date, k: int, calendar: list[date]
) -> tuple[dict[str, date], list[DateAdjustment]]:
    """The event's DATE_FIELDS moved with it, and the ones not moved by k trading days."""
    days = (new_day - event["first_tradeable_date"]).days
    shifted, adjustments = {}, []

    def adjust(field, value, method):
        adjustments.append(DateAdjustment(event_id=event["event_id"], field=field,
                                          original=event[field], shifted=value, method=method))
        shifted[field] = value

    for field in DATE_FIELDS:
        value = event[field]
        if value is None or pd.isna(value):
            continue
        moved = shift_trading_days(value, k, calendar)
        if moved is None:
            adjust(field, value + timedelta(days=days), "calendar_days")
        else:
            shifted[field] = moved
    if shifted.get("announcement_date", new_day) > new_day:
        adjust("announcement_date", new_day, "clamped")
    if "record_date" in shifted and shifted.get("payable_date", date.max) < shifted["record_date"]:
        adjust("payable_date", shifted["record_date"], "clamped")
    return shifted, adjustments


def first_tradeable_date(acceptance_et: datetime, calendar: list[date]) -> date:
    """store.prices.first_tradeable_date if implemented, else the data_schema.md rule."""
    try:
        return store_prices.first_tradeable_date(acceptance_et)
    except NotImplementedError:
        return local_first_tradeable_date(acceptance_et, calendar)


def _acceptance_on(new_day: date, original: datetime, calendar: list[date]) -> datetime:
    """original's clock time, on the day whose first tradeable date is new_day.

    Before the close: new_day itself. At or after the close: the previous trading day.
    """
    day = new_day
    if original.time() >= MARKET_CLOSE:
        day = calendar[bisect.bisect_left(calendar, new_day) - 1]
    return datetime.combine(day, original.time())


def valid_day_positions(
    dates: np.ndarray,
    is_bad: np.ndarray,
    real_positions: np.ndarray,
    min_gap_days: int,
    holding_days: int,
    first_usable: np.datetime64 | None = None,
) -> np.ndarray:
    """Positions (into one ticker's date-sorted rows) a placebo event may be dated on."""
    n = len(dates)
    bad = np.concatenate([[0], np.cumsum(is_bad, dtype=int)])  # bad[j] = bad rows before j
    pos = np.arange(n)
    ok = (pos >= 1) & (pos + holding_days <= n)
    ok[1:] &= ~is_bad[:-1]  # clean prior row
    end = np.minimum(pos + holding_days, n)
    ok &= bad[end] - bad[pos] == 0  # clean window
    if first_usable is not None:
        ok &= dates >= first_usable
    for r in real_positions:
        ok[max(r - min_gap_days + 1, 0) : max(r + min_gap_days, 0)] = False
    return np.flatnonzero(ok)


def make_placebo(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    seed: int,
    min_gap_days: int = DEFAULT_MIN_GAP_DAYS,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    usable_from: Mapping[str, date] | None = None,
) -> tuple[pd.DataFrame, PlaceboInfo]:
    """(events with eligible rows moved to placebo dates, info). Prices are not changed.

    usable_from (ticker -> first usable date, from universe_quality.csv) is only needed
    when prices still hold rows from before it; load_prices(usable_only=True) drops them.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")
    if min_gap_days < holding_days:
        raise ValueError(
            f"min_gap_days={min_gap_days} must be at least holding_days={holding_days}, "
            "or a placebo window could share days with a real event's window"
        )

    events = events.reset_index(drop=True)
    dates_all = pd.to_datetime(prices["date"]).dt.normalize()
    calendar = sorted({d.date() for d in dates_all.unique()})
    ticker_rows = prices.assign(_day=dates_all.to_numpy()).groupby("ticker", sort=True)
    real_days = pd.to_datetime(events["first_tradeable_date"]).dt.normalize()
    is_eligible = events.index.isin(eligible(events).index)

    out = events.copy()
    out["placebo"] = True
    out["original_first_tradeable_date"] = events["first_tradeable_date"]
    moved: dict[str, date] = {}
    valid_days: dict[str, int] = {}
    dropped: list[DroppedEvent] = []
    drop_index = []
    shifts: dict[str, int] = {}
    date_adjustments: list[DateAdjustment] = []

    tickers = sorted(events.loc[is_eligible, "ticker"].unique())
    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng([seed, _PLACEBO_STREAM, i])
        targets = events.index[is_eligible & (events["ticker"] == ticker).to_numpy()]
        if ticker in ticker_rows.groups:
            rows = ticker_rows.get_group(ticker).sort_values("_day")
        else:
            rows = prices.iloc[:0].assign(_day=pd.Series(dtype="datetime64[ns]"))
        dates = rows["_day"].to_numpy()
        real = real_days[(events["ticker"] == ticker).to_numpy()].to_numpy()
        first_usable = None
        if usable_from is not None and ticker in usable_from:
            first_usable = np.datetime64(pd.Timestamp(usable_from[ticker]))
        valid = valid_day_positions(
            dates,
            rows["is_bad_price"].to_numpy(dtype=bool),
            np.searchsorted(dates, real),
            min_gap_days,
            holding_days,
            first_usable,
        )
        valid_days[ticker] = len(valid)

        order = rng.permutation(len(targets))
        n_kept = min(len(targets), len(valid))
        chosen = np.sort(rng.choice(valid, size=n_kept, replace=False))
        new_positions = rng.permutation(chosen)
        mcap = rows["mcap_usd"].to_numpy(dtype=float)
        bucket = rows["mcap_bucket"].to_numpy()
        for k, j in enumerate(order):
            idx = targets[j]
            event_id = events.at[idx, "event_id"]
            if k >= n_kept:
                dropped.append(DroppedEvent(event_id=event_id, ticker=ticker,
                                            n_valid_days=len(valid)))
                drop_index.append(idx)
                continue
            p = int(new_positions[k])
            new_day = pd.Timestamp(dates[p]).date()
            acceptance = _acceptance_on(
                new_day, pd.Timestamp(events.at[idx, "acceptance_datetime"]).to_pydatetime(),
                calendar,
            )
            if first_tradeable_date(acceptance, calendar) != new_day:
                raise ValueError(f"{event_id}: acceptance {acceptance} does not map to {new_day}")
            out.at[idx, "first_tradeable_date"] = new_day
            out.at[idx, "acceptance_datetime"] = acceptance
            out.at[idx, "mcap_usd"] = mcap[p - 1]
            out.at[idx, "mcap_bucket"] = bucket[p - 1]
            if events.at[idx, "event_type"] == "buyback":
                amount = events.at[idx, "buyback_amount_usd"]
                if amount is not None and not pd.isna(amount):
                    out.at[idx, "buyback_pct_mcap"] = amount / mcap[p - 1] * 100
            moved[event_id] = events.at[idx, "first_tradeable_date"]

            # The other date fields move by as many trading days as the event did.
            shift = p - int(np.searchsorted(dates, real_days[idx].to_datetime64()))
            fields, adjustments = _shift_dates(events.loc[idx], new_day, shift, calendar)
            for field, value in fields.items():
                out.at[idx, field] = value
            shifts[event_id] = shift
            date_adjustments += adjustments

    out = out.drop(index=drop_index)
    out = out.sort_values(["first_tradeable_date", "ticker", "event_id"], ignore_index=True)
    info = PlaceboInfo(
        seed=seed,
        min_gap_days=min_gap_days,
        holding_days=holding_days,
        n_eligible=int(is_eligible.sum()),
        n_moved=len(moved),
        original_first_tradeable_date=moved,
        valid_days=valid_days,
        dropped=dropped,
        shift_trading_days=shifts,
        date_adjustments=date_adjustments,
    )
    return out, info


def make_placebo_universes(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    n: int,
    base_seed: int,
    min_gap_days: int = DEFAULT_MIN_GAP_DAYS,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    usable_from: Mapping[str, date] | None = None,
) -> list[tuple[int, pd.DataFrame, PlaceboInfo]]:
    """(seed, events_placebo, info) for seeds base_seed .. base_seed + n - 1."""
    universes = []
    for seed in range(base_seed, base_seed + n):
        placebo, info = make_placebo(prices, events, seed, min_gap_days, holding_days,
                                     usable_from)
        universes.append((seed, placebo, info))
    return universes
