"""Derived event fields, computed in code (never asked of the LLM).

- Duplicates: the same event in two filings (an 8-K and its 8-K/A, or a press release
  filed twice) keeps its first filing; later copies become is_new_announcement=False
  with a "duplicate_of:<event_id>" flag.
- Market data as of the trading day before first_tradeable_date: mcap_usd, mcap_bucket
  and buyback_pct_mcap.
- Dividend history: prev_dividend_per_share (the company's previous regular dividend) and
  dividend_change_pct for regular dividends.
- sector, from universe.csv.
"""

import pandas as pd

DUPLICATE_DAYS = {"dividend": 45, "buyback": 30}  # copies are filed within this many days


def _same(a, b) -> bool:
    """Equal, or one side missing (one filing states a date the other leaves out)."""
    if pd.isna(a) or pd.isna(b):
        return True
    if isinstance(a, float | int):
        return abs(float(a) - float(b)) <= 1e-6 * max(abs(float(a)), 1.0)
    return a == b


def _is_copy(row, first) -> bool:
    if row["event_type"] == "dividend":
        if pd.isna(row["dividend_per_share"]) or pd.isna(first["dividend_per_share"]):
            return False
        fields = ("dividend_per_share", "dividend_kind", "record_date", "payable_date")
    else:
        sizes = ("buyback_amount_usd", "buyback_shares")
        if not any(pd.notna(row[f]) and pd.notna(first[f]) for f in sizes):
            return False  # no size stated on both sides: can't tell
        fields = (*sizes, "is_new_program")
    return all(_same(row[f], first[f]) for f in fields)


def mark_duplicates(events: pd.DataFrame) -> pd.DataFrame:
    """Later copies of an already announced event become not new, flagged.

    A copy is a new event of the same company and type, filed within DUPLICATE_DAYS of
    an earlier one, whose amounts and dates match it (or are missing on one side).
    """
    events = events.sort_values(["first_tradeable_date", "acceptance_datetime", "event_id"])
    events = events.reset_index(drop=True)
    kept: dict[tuple, list[int]] = {}  # (ticker, type) -> rows kept as new
    for i, row in events.iterrows():
        if not row["is_new_announcement"]:
            continue
        group = kept.setdefault((row["ticker"], row["event_type"]), [])
        day = pd.Timestamp(row["first_tradeable_date"])
        limit = DUPLICATE_DAYS[row["event_type"]]
        first = next(
            (
                j
                for j in reversed(group)
                if (day - pd.Timestamp(events.at[j, "first_tradeable_date"])).days <= limit
                and _is_copy(row, events.loc[j])
            ),
            None,
        )
        if first is None:
            group.append(i)
            continue
        events.at[i, "is_new_announcement"] = False
        events.at[i, "flags"] = [*row["flags"], f"duplicate_of:{events.at[first, 'event_id']}"]
    return events


def add_market_data(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """mcap_usd and mcap_bucket on the last trading day before first_tradeable_date."""
    events = events.drop(columns=["mcap_usd", "mcap_bucket"], errors="ignore")
    left = events.assign(
        _ftd=pd.to_datetime(events["first_tradeable_date"]).astype("datetime64[ns]")
    )
    left = left.reset_index().sort_values("_ftd")
    left["ticker"] = left["ticker"].astype(object)
    right = prices[["ticker", "date", "mcap_usd", "mcap_bucket"]].dropna(subset=["mcap_usd"])
    right = right.assign(date=pd.to_datetime(right["date"]).astype("datetime64[ns]"))
    right = right.assign(ticker=right["ticker"].astype(object)).sort_values("date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="_ftd",
        right_on="date",
        by="ticker",
        allow_exact_matches=False,  # strictly before the event date
    )
    merged = merged.set_index("index").sort_index()
    events = events.assign(mcap_usd=merged["mcap_usd"], mcap_bucket=merged["mcap_bucket"])
    amount = events["buyback_amount_usd"].astype(float)
    events["buyback_pct_mcap"] = (amount / events["mcap_usd"] * 100).where(
        events["event_type"] == "buyback"
    )
    return events


def add_dividend_history(events: pd.DataFrame) -> pd.DataFrame:
    """prev_dividend_per_share and dividend_change_pct from earlier regular dividends.

    Only new, passed regular or initial dividends count as history. The change is set
    for regular dividends only.
    """
    events = events.copy()
    events["prev_dividend_per_share"] = None
    events["dividend_change_pct"] = None
    usable = (
        (events["event_type"] == "dividend")
        & events["is_new_announcement"].astype(bool)
        & (events["extraction_status"] == "passed")
        & events["dividend_kind"].isin(["regular", "initial"])
    )
    order = events[usable].sort_values(["first_tradeable_date", "acceptance_datetime"])
    for _, rows in order.groupby("ticker"):
        prev = None
        for i, row in rows.iterrows():
            dps = float(row["dividend_per_share"])
            if prev is not None:
                events.at[i, "prev_dividend_per_share"] = prev
                if row["dividend_kind"] == "regular" and prev > 0:
                    events.at[i, "dividend_change_pct"] = (dps / prev - 1) * 100
            prev = dps
    return events


def add_sector(events: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    sectors = universe.set_index("ticker")["sector"]
    return events.assign(sector=events["ticker"].map(sectors))


def derive(events: pd.DataFrame, prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """All derived fields, in order: duplicates first, so history skips copies."""
    events = mark_duplicates(events)
    events = add_market_data(events, prices)
    events = add_dividend_history(events)
    events = add_sector(events, universe)
    return events.sort_values(["first_tradeable_date", "ticker", "event_id"], ignore_index=True)
