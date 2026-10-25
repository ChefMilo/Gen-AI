from datetime import date

import numpy as np
import pandas as pd
import pytest

from evaluation.reference_car import event_cars
from universe_helpers import zero_noise
from universes.fake_events import MARKET_CLOSE, make_fake_events, rows_as_events
from universes.placebo import (
    PLACEBO_COLUMNS,
    first_tradeable_date,
    make_placebo,
    make_placebo_universes,
)
from universes.planted import PlantSpec, default_spec_grid, eligible, make_planted
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"
H = 5
GAP = 10
MIN_EVENTS = 20  # configs/default.yaml
EFFECT = 0.02

RECOMPUTED = ["first_tradeable_date", "acceptance_datetime", "mcap_usd", "mcap_bucket",
              "buyback_pct_mcap"]


@pytest.fixture(scope="module")
def benchmarks():
    return make_synthetic_benchmarks(START, END, seed=0)


@pytest.fixture(scope="module")
def prices(benchmarks):
    return zero_noise(make_synthetic_prices(TICKERS, START, END, seed=0), benchmarks)


@pytest.fixture(scope="module")
def events(prices):
    return make_fake_events(prices, seed=0)


@pytest.fixture(scope="module")
def placebo(prices, events):
    return make_placebo(prices, events, seed=0, min_gap_days=GAP, holding_days=H)


def ticker_rows(prices):
    """ticker -> (dates as datetime64, is_bad, mcap_usd, mcap_bucket), in date order."""
    out = {}
    for ticker, rows in prices.groupby("ticker"):
        rows = rows.sort_values("date")
        out[ticker] = (
            pd.to_datetime(rows["date"]).to_numpy(),
            rows["is_bad_price"].to_numpy(dtype=bool),
            rows["mcap_usd"].to_numpy(dtype=float),
            rows["mcap_bucket"].to_numpy(),
        )
    return out


def moved_rows(placebo_events, info):
    return placebo_events[placebo_events["event_id"].isin(info.original_first_tradeable_date)]


def test_counts_and_event_ids_kept(placebo, events):
    out, info = placebo
    dropped = {d.event_id for d in info.dropped}
    real = eligible(events)
    assert info.n_eligible == len(real)
    assert info.n_moved == len(real) - len(dropped) == len(eligible(out))
    assert set(out["event_id"]) == set(events["event_id"]) - dropped
    assert out["event_id"].is_unique

    kept = real[~real["event_id"].isin(dropped)]
    pd.testing.assert_series_equal(
        eligible(out).groupby("ticker").size(), kept.groupby("ticker").size()
    )


def test_ineligible_rows_unchanged(placebo, events):
    out, _ = placebo
    before = events[~events["event_id"].isin(eligible(events)["event_id"])]
    after = out.set_index("event_id").loc[before["event_id"]].reset_index()
    pd.testing.assert_frame_equal(
        after.drop(columns=PLACEBO_COLUMNS), before.reset_index(drop=True)
    )
    assert after["placebo"].all()
    assert (after["original_first_tradeable_date"] == after["first_tradeable_date"]).all()
    assert len(eligible(after)) == 0


def test_new_dates_are_valid_and_far_from_real_events(placebo, prices, events):
    out, info = placebo
    rows = ticker_rows(prices)
    moved = moved_rows(out, info)
    assert len(moved) > 0
    real_days = pd.to_datetime(events["first_tradeable_date"]).to_numpy()
    for ticker, group in moved.groupby("ticker"):
        dates, bad, _, _ = rows[ticker]
        real = np.searchsorted(dates, real_days[(events["ticker"] == ticker).to_numpy()])
        for day in pd.to_datetime(group["first_tradeable_date"]).to_numpy():
            p = int(np.searchsorted(dates, day))
            assert dates[p] == day  # a trading day of this ticker
            assert p >= 1 and not bad[p - 1]  # clean prior row for the mcap fields
            assert p + H <= len(dates) and not bad[p : p + H].any()  # full clean window
            assert np.abs(real - p).min() >= GAP
        # never two events on one day (the gap also keeps them off the ineligible rows)
        assert group["first_tradeable_date"].is_unique
    assert (moved["first_tradeable_date"] != moved["original_first_tradeable_date"]).all()
    assert moved["placebo"].all()
    originals = events.set_index("event_id").loc[moved["event_id"], "first_tradeable_date"]
    assert list(moved["original_first_tradeable_date"]) == list(originals)
    assert dict(zip(moved["event_id"], originals, strict=True)) == (
        info.original_first_tradeable_date
    )


def test_acceptance_moves_with_the_date(placebo, prices, events):
    out, info = placebo
    calendar = sorted({d.date() for d in pd.to_datetime(prices["date"]).unique()})
    moved = moved_rows(out, info)
    original = events.set_index("event_id").loc[moved["event_id"], "acceptance_datetime"]
    for acc, ftd, orig in zip(moved["acceptance_datetime"], moved["first_tradeable_date"],
                              original, strict=True):
        acc = pd.Timestamp(acc).to_pydatetime()
        assert acc.time() == pd.Timestamp(orig).time()
        assert first_tradeable_date(acc, calendar) == ftd
    # both branches of the rule are exercised
    after_close = [pd.Timestamp(a).time() >= MARKET_CLOSE for a in moved["acceptance_datetime"]]
    assert any(after_close) and not all(after_close)


def test_point_in_time_fields_from_prior_day(placebo, prices, events):
    out, info = placebo
    rows = ticker_rows(prices)
    moved = moved_rows(out, info)
    for _, event in moved.iterrows():
        dates, _, mcap, bucket = rows[event["ticker"]]
        p = int(np.searchsorted(dates, np.datetime64(event["first_tradeable_date"])))
        assert event["mcap_usd"] == mcap[p - 1]
        assert event["mcap_bucket"] == bucket[p - 1]
        if event["event_type"] == "buyback":
            assert np.isclose(event["buyback_pct_mcap"],
                              event["buyback_amount_usd"] / mcap[p - 1] * 100, rtol=1e-12)
        else:
            assert pd.isna(event["buyback_pct_mcap"])

    # Everything else, dividend_change_pct included, is exactly as extracted.
    before = events.set_index("event_id").loc[moved["event_id"]].drop(columns=RECOMPUTED)
    after = moved.set_index("event_id").drop(columns=RECOMPUTED + PLACEBO_COLUMNS)
    pd.testing.assert_frame_equal(after, before)


def test_still_validates_as_events(placebo):
    out, _ = placebo
    models = rows_as_events(out)
    assert len(models) == len(out)
    # Event ignores the extra columns; the original dates live in PlaceboInfo too.
    assert not any(hasattr(m, "placebo") for m in models)


def test_deterministic_per_seed(placebo, prices, events):
    again, again_info = make_placebo(prices, events, seed=0, min_gap_days=GAP, holding_days=H)
    pd.testing.assert_frame_equal(placebo[0], again)
    assert placebo[1] == again_info


def test_different_seeds_give_different_dates(placebo, prices, events):
    other, _ = make_placebo(prices, events, seed=1, min_gap_days=GAP, holding_days=H)
    a = placebo[0].set_index("event_id")["first_tradeable_date"]
    b = other.set_index("event_id").loc[a.index, "first_tradeable_date"]
    moved = list(placebo[1].original_first_tradeable_date)
    assert (a[moved] == b[moved]).mean() < 0.05


def test_requires_gap_at_least_holding_days(prices, events):
    with pytest.raises(ValueError, match="min_gap_days"):
        make_placebo(prices, events, seed=0, min_gap_days=H - 1, holding_days=H)
    make_placebo(prices, events, seed=0, min_gap_days=H, holding_days=H)


def test_too_few_valid_days_drops_and_reports(prices, events):
    # Half a year of prices and a wide gap leave fewer valid days than events.
    short = prices[pd.to_datetime(prices["date"]) < "2015-07-01"]
    short_events = events[pd.to_datetime(events["first_tradeable_date"]) < "2015-06-01"]
    out, info = make_placebo(short, short_events, seed=0, min_gap_days=40, holding_days=H)
    assert info.dropped
    for d in info.dropped:
        n_real = (eligible(short_events)["ticker"] == d.ticker).sum()
        assert d.n_valid_days == info.valid_days[d.ticker] < n_real
        assert d.event_id not in set(out["event_id"])
    kept = eligible(out)
    assert kept.groupby("ticker").size().to_dict() == {
        t: min(n, info.valid_days[t])
        for t, n in eligible(short_events).groupby("ticker").size().items()
        if min(n, info.valid_days[t]) > 0
    }
    assert not kept.duplicated(["ticker", "first_tradeable_date"]).any()
    assert info.n_moved + len(info.dropped) == info.n_eligible


def test_usable_from_is_respected(prices, events):
    usable_from = {t: date(2018, 1, 1) for t in TICKERS}
    out, info = make_placebo(prices, events, seed=0, min_gap_days=GAP, holding_days=H,
                             usable_from=usable_from)
    assert (moved_rows(out, info)["first_tradeable_date"] >= date(2018, 1, 1)).all()


def test_plant_on_real_events_never_reaches_placebo(placebo, prices, benchmarks, events):
    """The key property: an effect planted on every eligible real event reads back on the
    real events and is exactly zero on the placebo events."""
    planted = prices
    for event_type in ("dividend", "buyback"):
        spec = PlantSpec(spec_id=f"all_{event_type}", event_type=event_type, effect=EFFECT,
                         holding_days=H, strength="moderate", complexity="single",
                         feature_type="structured")
        planted, truth = make_planted(planted, events, spec, min_events=1)
        assert truth.n_targets == (eligible(events)["event_type"] == event_type).sum()

    out, info = placebo
    cars = event_cars(planted, benchmarks, eligible(out), H)
    assert cars.attrs["skipped"].empty  # every placebo event is measurable
    assert len(cars) == info.n_moved
    np.testing.assert_allclose(cars["car"], 0.0, rtol=0, atol=1e-9)

    # Real events read the effect back; where real windows overlap they add up.
    real = eligible(events)
    real_cars = event_cars(planted, benchmarks, real, H)
    assert (real_cars["car"] >= EFFECT - 1e-9).all()
    rows = ticker_rows(prices)
    pos = {
        e: int(np.searchsorted(rows[t][0], np.datetime64(d)))
        for e, t, d in zip(real["event_id"], real["ticker"], real["first_tradeable_date"],
                           strict=True)
    }
    by_ticker = real.groupby("ticker")["event_id"].apply(list).to_dict()
    isolated = [
        e for e, t in zip(real["event_id"], real["ticker"], strict=True)
        if all(abs(pos[e] - pos[o]) >= H for o in by_ticker[t] if o != e)
    ]
    clean = real_cars[real_cars["event_id"].isin(isolated)]
    assert len(clean) > len(real_cars) * 0.8
    np.testing.assert_allclose(clean["car"], EFFECT, rtol=0, atol=1e-9)


def test_make_placebo_universes(prices, events):
    universes = make_placebo_universes(prices, events, n=3, base_seed=7, min_gap_days=GAP,
                                       holding_days=H)
    assert [seed for seed, _, _ in universes] == [7, 8, 9]
    for seed, out, info in universes:
        assert info.seed == seed
        single, _ = make_placebo(prices, events, seed=seed, min_gap_days=GAP, holding_days=H)
        pd.testing.assert_frame_equal(out, single)


@pytest.mark.slow
def test_default_grid_on_full_placebo_universe(large_fake_prices, large_fake_events):
    out, info = make_placebo(large_fake_prices, large_fake_events, seed=0)
    real_specs, _ = default_spec_grid(large_fake_events, MIN_EVENTS)
    specs, dropped = default_spec_grid(out, MIN_EVENTS)
    assert len(specs) >= 0.9 * len(real_specs)
    assert {(s.complexity, s.feature_type) for s in specs} == {
        ("single", "structured"), ("conjunction", "structured"),
        ("single", "text"), ("conjunction", "text"),
    }
    assert {s.direction for s in specs} == {1, -1}
