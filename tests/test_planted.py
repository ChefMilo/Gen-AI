import numpy as np
import pandas as pd
import pytest

from evaluation.reference_car import CAR_COLUMNS, event_cars, summary
from universes.fake_events import make_fake_events
from universes.planted import (
    STRENGTHS,
    PlantSpec,
    default_spec_grid,
    eligible,
    make_planted,
    select_targets,
)
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"
H = 5
MIN_EVENTS = 20  # configs/default.yaml

# The default grid on the documented fake-events scale (PR #2: 180 tickers, 2015-2026).
GRID_TICKERS = [f"G{i:03d}" for i in range(180)]
GRID_START, GRID_END = "2015-01-01", "2026-12-31"
# Special dividends are about 3% of ticker-years, so special + small cap gives about 13.
KNOWN_SHORT = {f"dividend_special_small_{round(e * 10_000)}bp" for e in STRENGTHS}

SPEC = PlantSpec(
    spec_id="raise",
    event_type="dividend",
    condition="dividend_change_pct > 10",
    effect=0.02,
    holding_days=H,
    strength="moderate",
    complexity="single",
    feature_type="structured",
)


def zero_noise(prices, benchmarks):
    """Every stock's ret equals SPY's, so the reference AR is zero before planting."""
    spy = benchmarks[benchmarks["ticker"] == "SPY"].set_index("date")["ret"]
    out = prices.assign(ret=prices["date"].map(spy).to_numpy()).sort_values(["ticker", "date"])
    growth = (1 + out["ret"].fillna(0)).groupby(out["ticker"]).cumprod()
    first = out.groupby("ticker")["adj_close"].transform("first")
    return out.assign(adj_close=first * growth).sort_index()


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
def planted(prices, events):
    return make_planted(prices, events, SPEC, MIN_EVENTS)


def window_mask(prices, events, holding_days):
    """Boolean mask over prices rows inside any of the events' windows."""
    mask = pd.Series(False, index=prices.index)
    days = pd.to_datetime(events["first_tradeable_date"])
    for ticker, day0 in zip(events["ticker"], days, strict=True):
        rows = prices[prices["ticker"] == ticker].sort_values("date")
        pos = rows["date"].searchsorted(day0)
        mask[rows.index[pos : pos + holding_days]] = True
    return mask


def test_zero_noise_prices_have_zero_reference_car(prices, benchmarks, events):
    cars = event_cars(prices, benchmarks, eligible(events), H)
    assert len(cars) > 0
    np.testing.assert_allclose(cars["car"], 0.0, atol=1e-12)


def test_target_car_equals_planted_effect(planted, benchmarks, events):
    planted_prices, truth = planted
    overlapping = {e for o in truth.overlaps for e in o.event_ids}
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    cars = event_cars(planted_prices, benchmarks, targets, H)
    clean = cars[~cars["event_id"].isin(overlapping)]
    assert len(clean) >= MIN_EVENTS
    np.testing.assert_allclose(clean["car"], SPEC.effect, rtol=0, atol=1e-9)
    assert summary(clean)["hit_rate"] == 1.0


def test_non_target_car_unchanged(planted, prices, benchmarks, events):
    planted_prices, truth = planted
    others = eligible(events)
    others = others[~others["event_id"].isin(truth.target_event_ids)]
    # Windows of non-targets that touch a target window would move by design.
    touched = window_mask(prices, events[events["event_id"].isin(truth.target_event_ids)], H)
    clean = others[~others["event_id"].isin(_touching(prices, others, touched))]
    assert len(clean) > len(others) / 2
    before = event_cars(prices, benchmarks, clean, H)
    after = event_cars(planted_prices, benchmarks, clean, H)
    pd.testing.assert_frame_equal(before, after)


def _touching(prices, events, touched):
    """event_ids whose window includes a touched row."""
    ids = []
    days = pd.to_datetime(events["first_tradeable_date"])
    for event_id, ticker, day0 in zip(events["event_id"], events["ticker"], days, strict=True):
        rows = prices[prices["ticker"] == ticker].sort_values("date")
        pos = rows["date"].searchsorted(day0)
        if touched[rows.index[pos : pos + H]].any():
            ids.append(event_id)
    return ids


def test_adj_close_consistent_with_ret(planted):
    df = planted[0].sort_values(["ticker", "date"])
    expected = df.groupby("ticker")["adj_close"].pct_change()
    np.testing.assert_allclose(df["ret"], expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_rows_outside_windows_unchanged(planted, prices, events):
    planted_prices, truth = planted
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    inside = window_mask(prices, targets, H)
    assert inside.sum() == truth.n_targets * H - sum(o.n_days for o in truth.overlaps)

    # ret and every column but adj_close are identical outside the windows; benchmarks
    # are never touched (make_planted does not take them).
    pd.testing.assert_frame_equal(
        planted_prices.loc[~inside].drop(columns="adj_close"),
        prices.loc[~inside].drop(columns="adj_close"),
    )
    pd.testing.assert_frame_equal(
        planted_prices.drop(columns=["ret", "adj_close"]), prices.drop(columns=["ret", "adj_close"])
    )
    # adj_close only moves from a ticker's first window on (it carries the level shift).
    first_day = (
        pd.DataFrame({"ticker": targets["ticker"],
                      "day": pd.to_datetime(targets["first_tradeable_date"])})
        .groupby("ticker")["day"].min()
    )
    before = prices["date"] < prices["ticker"].map(first_day).fillna(pd.Timestamp.max)
    pd.testing.assert_series_equal(
        planted_prices.loc[before, "adj_close"], prices.loc[before, "adj_close"]
    )
    untouched_tickers = ~prices["ticker"].isin(first_day.index)
    pd.testing.assert_frame_equal(planted_prices[untouched_tickers], prices[untouched_tickers])


def test_targets_are_eligible_matches(planted, events):
    truth = planted[1]
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    assert truth.n_targets == len(truth.target_event_ids) == len(targets)
    assert targets["is_new_announcement"].all()
    assert (targets["extraction_status"] == "passed").all()
    assert (targets["event_type"] == "dividend").all()
    assert (targets["dividend_change_pct"] > 10).all()


def test_short_direction_lowers_car(prices, benchmarks, events):
    spec = SPEC.model_copy(update={"direction": -1})
    planted_prices, truth = make_planted(prices, events, spec, MIN_EVENTS)
    overlapping = {e for o in truth.overlaps for e in o.event_ids}
    targets = events[events["event_id"].isin(truth.target_event_ids)
                     & ~events["event_id"].isin(overlapping)]
    cars = event_cars(planted_prices, benchmarks, targets, H)
    np.testing.assert_allclose(cars["car"], -SPEC.effect, rtol=0, atol=1e-9)


def test_text_feature_spec(prices, events):
    spec = PlantSpec.model_validate({
        **SPEC.model_dump(), "spec_id": "framing", "condition": None, "feature_type": "text",
        "text_feature": {"feature": "dividend_framing", "equals": "fake_growth"},
    })
    targets = select_targets(events, spec)
    assert len(targets) > 0
    labels = targets["text_features"].map(lambda tf: tf["dividend_framing"]["label"])
    assert (labels == "fake_growth").all()


def test_deterministic(planted, prices, events):
    again_prices, again_truth = make_planted(prices, events, SPEC, MIN_EVENTS)
    pd.testing.assert_frame_equal(planted[0], again_prices)
    assert planted[1] == again_truth


def test_refuses_too_few_targets(prices, events):
    n = len(select_targets(events, SPEC))
    with pytest.raises(ValueError, match="fewer than min_events"):
        make_planted(prices, events, SPEC, min_events=n + 1)
    make_planted(prices, events, SPEC, min_events=n)


def test_overlapping_windows_apply_both(prices, benchmarks, events):
    # Two targets on one ticker, the second starting 2 trading days after the first.
    ticker = "T00"
    days = prices[prices["ticker"] == ticker].sort_values("date")["date"].iloc[500:503:2]
    base = eligible(events)
    base = base[base["event_type"] == "dividend"].iloc[[0, 1]].copy()
    base["ticker"] = ticker
    base["first_tradeable_date"] = [d.date() for d in days]
    base["event_id"] = ["ov-1", "ov-2"]
    spec = SPEC.model_copy(update={"spec_id": "overlap", "condition": None})

    planted_prices, truth = make_planted(prices, base, spec, min_events=2)
    assert len(truth.overlaps) == 1
    overlap = truth.overlaps[0]
    assert overlap.ticker == ticker and overlap.event_ids == ("ov-1", "ov-2")
    assert overlap.n_days == H - 2

    cars = event_cars(planted_prices, benchmarks, base, H).set_index("event_id")["car"]
    per_day = SPEC.effect / H
    # Each window gets its own effect plus the other's on the shared days.
    np.testing.assert_allclose(cars, SPEC.effect + per_day * (H - 2), rtol=0, atol=1e-9)
    window = planted_prices[(planted_prices["ticker"] == ticker)].sort_values("date")
    assert np.isclose(window["ret"].iloc[502] - prices.loc[window.index[502], "ret"], 2 * per_day)


def test_event_cars_skips_unmeasurable(prices, benchmarks, events):
    rows = eligible(events).iloc[:3].copy()
    last = prices[prices["ticker"] == rows["ticker"].iloc[0]]["date"].max()
    rows.iloc[0, rows.columns.get_loc("first_tradeable_date")] = last.date()
    rows.iloc[1, rows.columns.get_loc("first_tradeable_date")] = pd.Timestamp("2015-01-03").date()
    cars = event_cars(prices, benchmarks, rows, H)
    assert list(cars.columns) == CAR_COLUMNS
    skipped = cars.attrs["skipped"]
    reasons = dict(zip(skipped["event_id"], skipped["reason"], strict=True))
    assert reasons == {rows["event_id"].iloc[0]: "past_data", rows["event_id"].iloc[1]: "no_day0"}
    assert len(cars) == 1 and (cars["n_days"] == H).all()


def test_event_cars_skips_bad_price_windows(prices, benchmarks, events):
    rows = eligible(events).iloc[:1]
    ticker = rows["ticker"].iloc[0]
    day0 = pd.Timestamp(rows["first_tradeable_date"].iloc[0])
    bad = prices.copy()
    idx = bad[(bad["ticker"] == ticker) & (bad["date"] > day0)].sort_values("date").index[1]
    bad.loc[idx, "is_bad_price"] = True
    cars = event_cars(bad, benchmarks, rows, H)
    assert cars.empty and list(cars.attrs["skipped"]["reason"]) == ["bad_price"]


def test_summary():
    cars = pd.DataFrame({"car": [0.01, 0.03, -0.01, 0.01]})
    s = summary(cars)
    assert s["n"] == 4 and np.isclose(s["mean_car"], 0.01) and s["hit_rate"] == 0.75
    assert np.isclose(s["t_stat"], 0.01 / (np.std([0.01, 0.03, -0.01, 0.01], ddof=1) / 2))
    assert summary(cars.iloc[:0])["n"] == 0


def test_default_grid_shape():
    grid = default_spec_grid()
    assert len({s.spec_id for s in grid}) == len(grid)
    assert {s.strength for s in grid} == set(STRENGTHS.values())
    assert {(s.complexity, s.feature_type) for s in grid} == {
        ("single", "structured"), ("conjunction", "structured"),
        ("single", "text"), ("conjunction", "text"),
    }
    for s in grid:
        assert ("mcap_bucket" in (s.condition or "")) == (s.complexity == "conjunction")


@pytest.fixture(scope="module")
def grid_events():
    prices = make_synthetic_prices(GRID_TICKERS, GRID_START, GRID_END, seed=0)
    return make_fake_events(prices, seed=0)


def test_default_grid_has_enough_targets(grid_events):
    counts = {s.spec_id: len(select_targets(grid_events, s)) for s in default_spec_grid()}
    short = {spec_id for spec_id, n in counts.items() if n < MIN_EVENTS}
    assert short == KNOWN_SHORT, {k: counts[k] for k in short ^ KNOWN_SHORT}
