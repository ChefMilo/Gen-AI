import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from universes.fake_events import (
    FAKE_FILER,
    FAKE_QUOTE_PREFIX,
    first_tradeable_date,
    make_fake_events,
    rows_as_events,
)
from universes.synthetic import make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"


@pytest.fixture(scope="module")
def prices():
    return make_synthetic_prices(TICKERS, START, END, seed=0).assign(
        date=lambda df: df["date"].dt.date
    )


@pytest.fixture(scope="module")
def events(prices):
    return make_fake_events(prices, seed=0)


@pytest.fixture(scope="module")
def ticker_years(prices):
    return prices.groupby("ticker")["date"].agg(lambda d: len({x.year for x in d})).sum()


def test_every_row_is_a_valid_event(events):
    assert len(events) > 0
    assert len(rows_as_events(events)) == len(events)
    assert events["event_id"].is_unique


def test_same_seed_same_events(prices, events):
    pd.testing.assert_frame_equal(events, make_fake_events(prices, seed=0))
    assert not make_fake_events(prices, seed=1)["event_id"].equals(events["event_id"])


def test_years_filter_keeps_the_same_rows(prices, events):
    some = make_fake_events(prices, seed=0, years=[2017, 2018])
    assert {d.year for d in some["first_tradeable_date"]} == {2017, 2018}
    expected = events[events["first_tradeable_date"].map(lambda d: d.year in (2017, 2018))]
    pd.testing.assert_frame_equal(some, expected.reset_index(drop=True))


def test_first_tradeable_date_rule():
    calendar = [date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8)]
    assert first_tradeable_date(datetime(2024, 7, 3, 15, 59), calendar) == date(2024, 7, 3)
    assert first_tradeable_date(datetime(2024, 7, 3, 16, 0), calendar) == date(2024, 7, 5)
    assert first_tradeable_date(datetime(2024, 7, 4, 9, 0), calendar) == date(2024, 7, 5)
    assert first_tradeable_date(datetime(2024, 7, 6, 9, 0), calendar) == date(2024, 7, 8)


def test_event_dates_follow_prices(prices, events):
    calendar = sorted(prices["date"].unique())
    rows = prices.set_index(["ticker", "date"])
    for e in events.itertuples():
        ftd = e.first_tradeable_date
        assert first_tradeable_date(e.acceptance_datetime.to_pydatetime(), calendar) == ftd
        assert not rows.loc[(e.ticker, ftd), "is_bad_price"]
        ticker_rows = rows.loc[e.ticker]
        prior = ticker_rows[ticker_rows.index < ftd].iloc[-1]
        assert e.mcap_usd == prior["mcap_usd"]
        assert e.mcap_bucket == prior["mcap_bucket"]


def test_acceptance_times_cover_before_and_after_close(events):
    before_close = events["acceptance_datetime"].dt.hour < 16
    same_day = events["acceptance_datetime"].dt.date == events["first_tradeable_date"]
    assert (before_close & same_day).any()
    assert (~same_day).any()
    assert (same_day <= before_close).all()


def test_event_ids_and_filing_links(events):
    pattern = re.compile(rf"^{FAKE_FILER}-\d{{2}}-\d{{6}}$")
    assert events["accession_no"].map(lambda a: bool(pattern.match(a))).all()
    for e in events.itertuples():
        assert re.fullmatch(rf"{e.accession_no}-{e.event_type}-\d+", e.event_id)
        assert e.accession_no.replace("-", "") in e.source_url
    # One filing can hold a dividend and a buyback.
    assert (events.groupby("accession_no")["event_type"].nunique() > 1).any()


def test_dividend_fields(events, ticker_years):
    div = events[events["event_type"] == "dividend"]
    assert div["dividend_per_share"].between(0.001, 50).all()
    assert (div["record_date"] <= div["payable_date"]).all()
    assert div["ex_date"].isna().all()

    kinds = div["dividend_kind"].value_counts()
    assert set(kinds.index) <= {"regular", "special", "initial"}
    assert kinds["regular"] / len(div) > 0.95
    assert kinds.get("special", 0) > 0 and kinds.get("initial", 0) > 0

    has_prev = div.dropna(subset=["prev_dividend_per_share"])
    np.testing.assert_allclose(
        has_prev["dividend_change_pct"],
        (has_prev["dividend_per_share"] / has_prev["prev_dividend_per_share"] - 1) * 100,
    )
    assert div.loc[div["prev_dividend_per_share"].isna(), "dividend_change_pct"].isna().all()

    change = has_prev["dividend_change_pct"]
    assert (change == 0).mean() > 0.6
    assert 0.5 < (change > 0).sum() / ticker_years < 1.5
    assert 0 < (change < 0).mean() < 0.05


def test_prev_dividend_is_previous_new_dividend(events):
    series = events[
        events["is_new_announcement"] & events["dividend_kind"].isin(["regular", "initial"])
    ].sort_values(["ticker", "first_tradeable_date"])
    expected = series.groupby("ticker")["dividend_per_share"].shift(1)
    np.testing.assert_allclose(series["prev_dividend_per_share"], expected, equal_nan=True)


def test_buyback_fields(events):
    bb = events[events["event_type"] == "buyback"]
    assert (bb["buyback_amount_usd"] > 0).all()
    np.testing.assert_allclose(
        bb["buyback_pct_mcap"], bb["buyback_amount_usd"] / bb["mcap_usd"] * 100
    )
    assert set(bb["is_new_program"]) == {True, False}
    assert (
        bb["buyback_expiry"].dropna() > bb.loc[bb["buyback_expiry"].notna(), "first_tradeable_date"]
    ).all()


def test_event_counts_are_plausible(events, ticker_years):
    new = events[events["is_new_announcement"]]
    per_year = new.groupby("event_type").size() / ticker_years
    assert 3.5 <= per_year["dividend"] <= 4.5
    assert 0.3 <= per_year["buyback"] <= 1.5


def test_flagged_and_not_new_rows(events):
    flagged = events["extraction_status"] == "flagged"
    assert 0.01 <= flagged.mean() <= 0.1
    assert 0.01 <= (~events["is_new_announcement"]).mean() <= 0.1
    for e in events[flagged].itertuples():
        (flag,) = e.flags
        assert flag.startswith("missing_evidence:")
        assert flag.split(":", 1)[1] not in e.evidence
    assert (events.loc[~flagged, "flags"].map(len) == 0).all()
    usable = events[events["is_new_announcement"] & ~flagged]
    assert 0.8 < len(usable) / len(events) < 0.99


def test_quotes_and_labels_are_marked_fake(events):
    for e in events.itertuples():
        assert all(q.startswith(FAKE_QUOTE_PREFIX) for q in e.evidence.values())
        feature = "dividend_framing" if e.event_type == "dividend" else "buyback_rationale"
        assert set(e.text_features) == {feature}
        assert e.text_features[feature]["label"].startswith("fake_")
        assert e.text_features[feature]["quote"].startswith(FAKE_QUOTE_PREFIX)


def with_nan_mcap(prices, ticker_rows):
    """prices with mcap_usd and mcap_bucket NaN on the given (ticker -> row slice) rows."""
    out = prices.copy()
    for ticker, rows in ticker_rows.items():
        idx = out[out["ticker"] == ticker].sort_values("date").index[rows]
        out.loc[idx, ["mcap_usd", "mcap_bucket"]] = np.nan
    return out


def test_nan_mcap_rows_are_skipped(prices):
    # Like real prices: mcap missing at the start of history, and in a few odd rows.
    nan_prices = with_nan_mcap(prices, {"T00": slice(0, 400), "T01": slice(None, None, 3),
                                        "T02": slice(None)})
    events = make_fake_events(nan_prices, seed=0)
    assert len(rows_as_events(events)) == len(events)
    assert events["mcap_usd"].notna().all()
    assert set(events["mcap_bucket"]) <= {"small", "mid", "large"}
    assert "T02" not in set(events["ticker"])
    # Every event's mcap fields come from a prior row that has them.
    for ticker in ("T00", "T01"):
        rows = nan_prices[nan_prices["ticker"] == ticker].sort_values("date")
        dates = list(rows["date"])
        mcap = rows["mcap_usd"].to_numpy()
        mine = events[events["ticker"] == ticker]
        assert len(mine) > 0
        for day, value in zip(mine["first_tradeable_date"], mine["mcap_usd"], strict=True):
            assert mcap[dates.index(day) - 1] == value
    buybacks = events[events["event_type"] == "buyback"]
    assert buybacks["buyback_pct_mcap"].notna().all()
