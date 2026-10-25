from datetime import UTC, date, datetime

import pandas as pd
import pytest

from store import prices

# Wed 3 Jul 2024, then the 4 Jul holiday, Fri 5, weekend, Mon 8.
CALENDAR = [
    date(2024, 7, 1),
    date(2024, 7, 2),
    date(2024, 7, 3),
    date(2024, 7, 5),
    date(2024, 7, 8),
]


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A tiny Gen AI/Data folder: A keep, B keep_from 2024-07-03, C drop."""
    days = pd.to_datetime(CALENDAR)
    rows = []
    for ticker in ("A", "B", "C"):
        for d in days:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "adj_close": 10.0,
                    "ret": 0.0,
                    "is_bad_price": False,
                    "bad_price_reason": None,
                }
            )
    df = pd.DataFrame(rows)
    bad = (df["ticker"] == "A") & (df["date"] == pd.Timestamp("2024-07-02"))
    df.loc[bad, ["is_bad_price", "bad_price_reason"]] = [True, "extreme_return"]

    (tmp_path / "prices").mkdir()
    df.to_parquet(tmp_path / prices.PRICES)
    bench = pd.DataFrame(
        [
            {"date": d, "ticker": t, "adj_close": 1.0, "ret": 0.0}
            for t in ("MDY", "SPY")
            for d in reversed(days)
        ]
    )
    bench.to_parquet(tmp_path / prices.BENCHMARKS)
    pd.DataFrame({"date": CALENDAR}).to_csv(tmp_path / prices.TRADING_CALENDAR, index=False)
    pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "recommendation": ["keep", "keep_from", "drop"],
            "usable_from": ["2024-07-01", "2024-07-03", None],
        }
    ).to_csv(tmp_path / prices.UNIVERSE_QUALITY, index=False)

    monkeypatch.setenv(prices.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


def test_load_prices_drops_bad_days_dropped_companies_and_early_rows(data):
    df = prices.load_prices()
    assert set(df["ticker"]) == {"A", "B"}
    assert pd.Timestamp("2024-07-02") not in set(df.loc[df["ticker"] == "A", "date"])
    assert df.loc[df["ticker"] == "B", "date"].min() == pd.Timestamp("2024-07-03")
    assert not df["is_bad_price"].any()
    assert list(df[["ticker", "date"]].itertuples(index=False)) == sorted(
        df[["ticker", "date"]].itertuples(index=False)
    )


def test_load_prices_switches_and_filters(data):
    everything = prices.load_prices(drop_bad=False, usable_only=False)
    assert len(everything) == 3 * len(CALENDAR)
    a = prices.load_prices(tickers=["A"], start="2024-07-03", end=date(2024, 7, 5))
    assert list(a["date"]) == list(pd.to_datetime([date(2024, 7, 3), date(2024, 7, 5)]))
    assert prices.load_prices(tickers=[]).empty


def test_load_benchmarks_one_ticker_in_date_order(data):
    spy = prices.load_benchmarks()
    assert set(spy["ticker"]) == {"SPY"}
    assert spy["date"].is_monotonic_increasing
    assert set(prices.load_benchmarks("MDY")["ticker"]) == {"MDY"}
    with pytest.raises(ValueError):
        prices.load_benchmarks("STI")


@pytest.mark.parametrize(
    ("accepted", "expected"),
    [
        (datetime(2024, 7, 3, 15, 59), date(2024, 7, 3)),  # before the close
        (datetime(2024, 7, 3, 16, 0), date(2024, 7, 5)),  # at the close -> next trading day
        (datetime(2024, 7, 4, 9, 0), date(2024, 7, 5)),  # holiday
        (datetime(2024, 7, 6, 9, 0), date(2024, 7, 8)),  # Saturday
        (datetime(2024, 7, 3, 19, 59, tzinfo=UTC), date(2024, 7, 3)),  # 15:59 EDT
        (datetime(2024, 7, 3, 20, 0, tzinfo=UTC), date(2024, 7, 5)),  # 16:00 EDT
    ],
)
def test_first_tradeable_date_rule(accepted, expected):
    assert prices.first_tradeable_date(accepted, CALENDAR) == expected


def test_first_tradeable_date_reads_calendar_file(data):
    assert prices.trading_calendar() == CALENDAR
    assert prices.first_tradeable_date(datetime(2024, 7, 3, 17, 0)) == date(2024, 7, 5)


def test_first_tradeable_date_past_calendar_end():
    with pytest.raises(ValueError):
        prices.first_tradeable_date(datetime(2024, 7, 8, 16, 30), CALENDAR)
