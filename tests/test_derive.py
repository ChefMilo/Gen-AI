from datetime import date, datetime

import pandas as pd
import pytest

from ingest import derive
from store.schema import Event


def _event(n, ftd, event_type="dividend", ticker="A", **fields):
    base = dict(
        event_id=f"acc{n}-{event_type}-1",
        event_type=event_type,
        ticker=ticker,
        cik="1",
        company_name="A Co",
        accession_no=f"acc{n}",
        form_type="8-K",
        items="8.01",
        source_url="",
        acceptance_datetime=datetime.combine(ftd, datetime.min.time()),
        first_tradeable_date=ftd,
        is_new_announcement=True,
        extraction_status="passed",
        extracted_at=datetime(2026, 1, 1),
        extractor_version="test",
    )
    if event_type == "dividend":
        base |= {"dividend_per_share": 0.20, "dividend_kind": "regular"}
    else:
        base |= {"buyback_amount_usd": 10e6, "is_new_program": True}
    return Event(**(base | fields)).model_dump()


def _frame(*events):
    return pd.DataFrame(list(events), columns=list(Event.model_fields))


def test_mark_duplicates_keeps_the_first_filing():
    original = _event(1, date(2024, 1, 5), record_date=date(2024, 2, 1))
    amended = _event(2, date(2024, 1, 8), record_date=date(2024, 2, 1))
    next_quarter = _event(3, date(2024, 4, 5), record_date=date(2024, 5, 1))
    out = derive.mark_duplicates(_frame(amended, original, next_quarter))
    by_id = out.set_index("event_id")
    assert by_id.loc["acc1-dividend-1", "is_new_announcement"]
    assert not by_id.loc["acc2-dividend-1", "is_new_announcement"]
    assert by_id.loc["acc2-dividend-1", "flags"] == ["duplicate_of:acc1-dividend-1"]
    assert by_id.loc["acc3-dividend-1", "is_new_announcement"]


def test_mark_duplicates_buybacks_only_close_in_time():
    first = _event(1, date(2024, 1, 5), "buyback")
    copy = _event(2, date(2024, 1, 20), "buyback")
    next_year = _event(3, date(2025, 1, 5), "buyback")
    out = derive.mark_duplicates(_frame(first, copy, next_year)).set_index("event_id")
    assert out["is_new_announcement"].tolist() == [True, False, True]


def test_add_market_data_uses_the_day_before():
    prices = pd.DataFrame(
        {
            "ticker": ["A"] * 3,
            "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "mcap_usd": [100e6, 200e6, 300e6],
            "mcap_bucket": ["small", "mid", "large"],
        }
    )
    events = _frame(
        _event(1, date(2024, 1, 5), "buyback"),
        _event(2, date(2024, 1, 3)),  # no earlier price
    )
    out = derive.add_market_data(events, prices)
    assert out.loc[0, "mcap_usd"] == 200e6 and out.loc[0, "mcap_bucket"] == "mid"
    assert out.loc[0, "buyback_pct_mcap"] == pytest.approx(5.0)
    assert pd.isna(out.loc[1, "mcap_usd"]) and pd.isna(out.loc[1, "buyback_pct_mcap"])


def test_add_dividend_history_skips_specials_and_flagged():
    events = _frame(
        _event(1, date(2024, 1, 5), dividend_per_share=0.20),
        _event(2, date(2024, 2, 5), dividend_per_share=1.00, dividend_kind="special"),
        _event(3, date(2024, 3, 5), dividend_per_share=0.30, extraction_status="flagged"),
        _event(4, date(2024, 4, 5), dividend_per_share=0.22),
    )
    out = derive.add_dividend_history(events).set_index("event_id")
    assert out.loc["acc1-dividend-1", "prev_dividend_per_share"] is None
    assert out.loc["acc4-dividend-1", "prev_dividend_per_share"] == 0.20
    assert out.loc["acc4-dividend-1", "dividend_change_pct"] == pytest.approx(10.0)
    assert out.loc["acc2-dividend-1", "prev_dividend_per_share"] is None


def test_derive_adds_sector_and_sorts():
    prices = pd.DataFrame(columns=["ticker", "date", "mcap_usd", "mcap_bucket"])
    universe = pd.DataFrame({"ticker": ["A"], "sector": ["Energy"]})
    events = _frame(_event(2, date(2024, 2, 5)), _event(1, date(2024, 1, 5)))
    out = derive.derive(events, prices, universe)
    assert out["sector"].tolist() == ["Energy", "Energy"]
    assert out["event_id"].tolist() == ["acc1-dividend-1", "acc2-dividend-1"]
