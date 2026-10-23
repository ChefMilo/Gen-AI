"""Fake event store rows (events.parquet) built on a prices frame.

Every row validates against store.schema.Event. Filings, quotes and text features
are fake: accession numbers use the filer prefix FAKE_FILER, quotes start with
FAKE_QUOTE_PREFIX and text-feature labels start with "fake_". Final labels are
still to come from Retrieval (docs/data_schema.md, open questions).

Use rows_as_events() to turn the frame back into Event models.
"""

import bisect
import math
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from store.schema import Event, TextFeature

FAKE_FILER = "0009999999"
FAKE_QUOTE_PREFIX = "[FAKE]"
EXTRACTOR_VERSION = "fake-0.1"
EXTRACTED_AT = datetime(2026, 1, 1, 12, 0)
MARKET_CLOSE = time(16, 0)

DIVIDEND_FRAMING_LABELS = ("fake_routine", "fake_growth", "fake_cautious")
BUYBACK_RATIONALE_LABELS = ("fake_undervalued", "fake_excess_cash", "fake_offset_dilution")

FLAGGED_SHARE = 0.04
NOT_NEW_SHARE = 0.04

_EVENT_STREAM = 3


def _fake(text: str) -> str:
    return f"{FAKE_QUOTE_PREFIX} {text}"


def first_tradeable_date(acceptance_et: datetime, calendar: list[date]) -> date:
    """Same day if accepted before 16:00 ET on a trading day, else the next trading day.

    Local stand-in for store.prices.first_tradeable_date, which is still a stub.
    """
    day = acceptance_et.date()
    i = bisect.bisect_left(calendar, day)
    if i < len(calendar) and calendar[i] == day and acceptance_et.time() < MARKET_CLOSE:
        return day
    i = bisect.bisect_right(calendar, day)
    if i >= len(calendar):
        raise ValueError(f"no trading day after {acceptance_et}")
    return calendar[i]


def _acceptance(rng: np.random.Generator, ftd: date, calendar: list[date]) -> datetime:
    """An acceptance time whose first tradeable date is ftd."""
    prev_day = calendar[bisect.bisect_left(calendar, ftd) - 1]
    r = rng.random()
    if r < 0.6:  # during or before the session on ftd
        return datetime.combine(ftd, time(int(rng.integers(6, 16)), int(rng.integers(0, 60))))
    gap = [prev_day + timedelta(days=k) for k in range(1, (ftd - prev_day).days)]
    if r < 0.9 or not gap:  # after the close on the previous trading day
        return datetime.combine(prev_day, time(int(rng.integers(16, 22)), int(rng.integers(0, 60))))
    day = gap[int(rng.integers(len(gap)))]  # on a weekend or holiday
    return datetime.combine(day, time(int(rng.integers(0, 24)), int(rng.integers(0, 60))))


class _Ticker:
    """One ticker's price rows, and the positions an event may be dated on."""

    def __init__(self, rows: pd.DataFrame, calendar: list[date]):
        self.dates = list(rows["date"])
        self.mcap = rows["mcap_usd"].to_numpy()
        self.bucket = rows["mcap_bucket"].to_numpy()
        bad = rows["is_bad_price"].to_numpy(dtype=bool)
        cal_pos = {d: k for k, d in enumerate(calendar)}
        # An event day needs a clean row, a clean prior row (for mcap) and a calendar
        # day before it (for the after-close acceptance rule).
        self.eligible = np.array(
            [
                i >= 1 and not bad[i] and not bad[i - 1] and cal_pos[d] >= 1
                for i, d in enumerate(self.dates)
            ],
            dtype=bool,
        )

    def snap(self, pos: int) -> int | None:
        """The first eligible position at or after pos."""
        later = np.flatnonzero(self.eligible[max(pos, 0) :])
        return None if len(later) == 0 else max(pos, 0) + int(later[0])


def _event_fields(t: _Ticker, pos: int) -> dict:
    return {
        "first_tradeable_date": t.dates[pos],
        "mcap_usd": float(t.mcap[pos - 1]),
        "mcap_bucket": str(t.bucket[pos - 1]),
    }


def _dividend(rng, t: _Ticker, pos: int, dps: float, kind: str, prev: float | None) -> dict:
    ftd = t.dates[pos]
    record = ftd + timedelta(days=int(rng.integers(10, 21)))
    payable = record + timedelta(days=int(rng.integers(7, 21)))
    change = None if prev is None else (dps / prev - 1) * 100
    if kind == "special":
        framing = "fake_routine"
    elif change is not None and change > 0:
        framing = "fake_growth"
    elif change is not None and change < 0:
        framing = "fake_cautious"
    else:
        framing = "fake_routine"
    return {
        **_event_fields(t, pos),
        "event_type": "dividend",
        "dividend_per_share": dps,
        "dividend_kind": kind,
        "record_date": record,
        "payable_date": payable,
        "prev_dividend_per_share": prev,
        "dividend_change_pct": change,
        "text_features": {
            "dividend_framing": TextFeature(
                label=framing, quote=_fake(f"our {kind} dividend reflects {framing[5:]} policy")
            )
        },
        "evidence": {
            "dividend_per_share": _fake(f"a {kind} dividend of ${dps:.2f} per share"),
            "dividend_kind": _fake(f"declared a {kind} cash dividend"),
            "record_date": _fake(f"to shareholders of record on {record:%B %d, %Y}"),
            "payable_date": _fake(f"payable on {payable:%B %d, %Y}"),
        },
    }


def _buyback(rng, t: _Ticker, pos: int) -> dict:
    fields = _event_fields(t, pos)
    amount = max(1e6, round(fields["mcap_usd"] * rng.uniform(0.01, 0.08) / 1e6) * 1e6)
    is_new = bool(rng.random() < 0.6)
    expiry = None
    if rng.random() < 0.5:
        expiry = fields["first_tradeable_date"] + timedelta(days=int(365 * rng.integers(1, 4)))
    rationale = BUYBACK_RATIONALE_LABELS[int(rng.integers(len(BUYBACK_RATIONALE_LABELS)))]
    evidence = {
        "buyback_amount_usd": _fake(f"repurchase up to ${amount / 1e6:,.0f} million"),
        "is_new_program": _fake("a new share repurchase program" if is_new else "an increase"),
    }
    if expiry is not None:
        evidence["buyback_expiry"] = _fake(f"expires on {expiry:%B %d, %Y}")
    return {
        **fields,
        "event_type": "buyback",
        "buyback_amount_usd": amount,
        "is_new_program": is_new,
        "buyback_expiry": expiry,
        "buyback_pct_mcap": amount / fields["mcap_usd"] * 100,
        "text_features": {
            "buyback_rationale": TextFeature(
                label=rationale, quote=_fake(f"we are buying back stock: {rationale[5:]}")
            )
        },
        "evidence": evidence,
    }


def _ticker_events(rng, t: _Ticker) -> list[tuple[int, list[dict]]]:
    """(position, events) per filing for one ticker, before noise and accession numbers."""
    n = len(t.dates)
    filings: list[tuple[int, list[dict]]] = []

    # Regular dividends about every 63 trading days; ~5% of tickers initiate mid-sample.
    initiator = rng.random() < 0.05
    pos = int(rng.integers(0, n // 2 if initiator else 63)) if n > 63 else 0
    dps = round(float(rng.uniform(0.05, 1.0)), 2)
    prev = None
    kind = "initial" if initiator else "regular"
    while (p := t.snap(pos)) is not None:
        filings.append((p, [_dividend(rng, t, p, dps, kind, prev)]))
        prev, kind = dps, "regular"
        u = rng.random()
        if u < 0.25:
            dps *= 1 + rng.uniform(0.03, 0.15)
        elif u < 0.27:
            dps *= rng.uniform(0.5, 0.8)
        dps = max(0.01, round(dps, 2))
        pos = p + 63 + int(rng.integers(-4, 5))

    years = sorted({d.year for d in t.dates})
    for year in years:
        in_year = [i for i, d in enumerate(t.dates) if d.year == year and t.eligible[i]]
        if not in_year:
            continue
        if rng.random() < 0.03:
            p = in_year[int(rng.integers(len(in_year)))]
            amount = round(float(rng.uniform(0.5, 3.0)), 2)
            filings.append((p, [_dividend(rng, t, p, amount, "special", None)]))
        for _ in range(min(int(rng.poisson(0.8)), 2)):
            same_year = [f for f in filings if t.dates[f[0]].year == year]
            if same_year and rng.random() < 0.2:  # announced with a dividend
                p, events = same_year[int(rng.integers(len(same_year)))]
                if not any(e["event_type"] == "buyback" for e in events):
                    events.append(_buyback(rng, t, p))
                    continue
            p = in_year[int(rng.integers(len(in_year)))]
            filings.append((p, [_buyback(rng, t, p)]))
    return filings


def _add_noise(rng, t: _Ticker, filings: list[tuple[int, list[dict]]]) -> None:
    """Flag some events, and add filings that only report past activity."""
    for _, events in list(filings):
        for event in events:
            if rng.random() < FLAGGED_SHARE:
                field = sorted(event["evidence"])[int(rng.integers(len(event["evidence"])))]
                del event["evidence"][field]
                event["extraction_status"] = "flagged"
                event["flags"] = [f"missing_evidence:{field}"]
    for p, events in list(filings):
        for event in events:
            if rng.random() >= NOT_NEW_SHARE:
                continue
            later = t.snap(p + int(rng.integers(20, 61)))
            if later is None:
                continue
            copy = {
                **event,
                **_event_fields(t, later),
                "is_new_announcement": False,
                "extraction_status": "passed",
                "flags": [],
                "evidence": dict(event["evidence"]),
            }
            if copy["event_type"] == "dividend":
                copy["prev_dividend_per_share"] = None
                copy["dividend_change_pct"] = None
            else:
                copy["buyback_pct_mcap"] = copy["buyback_amount_usd"] / copy["mcap_usd"] * 100
            filings.append((later, [copy]))


def make_fake_events(
    prices: pd.DataFrame, seed: int, years: Iterable[int] | None = None
) -> pd.DataFrame:
    """Fake events.parquet rows for every ticker in prices.

    years keeps only events whose first_tradeable_date falls in those calendar
    years; the kept rows are the same as in the unfiltered output.
    """
    prices = prices.assign(date=pd.to_datetime(prices["date"]).dt.date)
    calendar = sorted(prices["date"].unique())
    tickers = sorted(prices["ticker"].unique())

    raw = []  # (acceptance, ticker, cik, events)
    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng([seed, _EVENT_STREAM, i])
        rows = prices[prices["ticker"] == ticker].sort_values("date")
        t = _Ticker(rows, calendar)
        filings = _ticker_events(rng, t)
        _add_noise(rng, t, filings)
        cik = f"99{i:08d}"
        for p, events in filings:
            raw.append((_acceptance(rng, t.dates[p], calendar), ticker, cik, events))

    raw.sort(key=lambda f: (f[0], f[1]))
    models = []
    seq_by_year: dict[int, int] = {}
    for acceptance, ticker, cik, events in raw:
        yy = acceptance.year % 100
        seq_by_year[yy] = seq_by_year.get(yy, 0) + 1
        accession = f"{FAKE_FILER}-{yy:02d}-{seq_by_year[yy]:06d}"
        items = "2.02,9.01" if not events[0].get("is_new_announcement", True) else "8.01,9.01"
        counters: dict[str, int] = {}
        for event in events:
            kind = event["event_type"]
            counters[kind] = counters.get(kind, 0) + 1
            event = {
                "is_new_announcement": True,
                "extraction_status": "passed",
                "flags": [],
                **event,
            }
            models.append(
                Event(
                    **event,
                    event_id=f"{accession}-{kind}-{counters[kind]}",
                    ticker=ticker,
                    cik=cik,
                    company_name=f"{ticker} Fake Corp",
                    accession_no=accession,
                    form_type="8-K",
                    items=items,
                    source_url=(
                        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                        f"{accession.replace('-', '')}/"
                    ),
                    acceptance_datetime=acceptance,
                    announcement_date=acceptance.date(),
                    extracted_at=EXTRACTED_AT,
                    extractor_version=EXTRACTOR_VERSION,
                )
            )

    if years is not None:
        keep = set(years)
        models = [m for m in models if m.first_tradeable_date.year in keep]
    if not models:
        return pd.DataFrame(columns=list(Event.model_fields))
    events = pd.DataFrame([m.model_dump() for m in models], columns=list(Event.model_fields))
    return events.sort_values(["first_tradeable_date", "ticker", "event_id"], ignore_index=True)


def rows_as_events(events: pd.DataFrame) -> list[Event]:
    """Validate every row of an events frame as an Event (NaN becomes None)."""

    def clean(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        if value is pd.NaT:
            return None
        return value

    return [
        Event.model_validate({k: clean(v) for k, v in row.items()})
        for row in events.to_dict(orient="records")
    ]
