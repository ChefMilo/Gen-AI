"""plant_universe: clashes between targets of different specs, spillover and baselines."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from evaluation.reference_car import event_cars
from evaluation.universe_record import (
    UniverseRecord,
    find_clashes,
    plant_universe,
    recompute_injected,
)
from universes.planted import PlantSpec, make_planted

H = 5
DAYS = pd.bdate_range("2020-01-01", periods=40)


def spec(spec_id: str, condition: str) -> PlantSpec:
    return PlantSpec(
        spec_id=spec_id, event_type="dividend", condition=condition, effect=0.02,
        holding_days=H, strength="moderate", complexity="single", feature_type="structured",
    )


SPECS = [spec("x", "x > 0"), spec("y", "y > 0")]


@pytest.fixture
def prices():
    rows = [
        {"ticker": t, "date": d, "ret": 0.0, "adj_close": 100.0, "is_bad_price": False}
        for t in ("A", "B")
        for d in DAYS
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def events():
    rows = [
        # event_id, ticker, day position, x, y
        ("e1", "A", 0, 1, 0),  # window 0-4, clashes with e2 (y, 2-6)
        ("e2", "A", 2, 0, 1),
        ("e3", "A", 33, 1, 1),  # targeted by both specs
        ("e4", "B", 2, 0, 1),  # same days as e1 but another ticker
        ("e5", "A", 8, 1, 0),  # window 8-12, ends the day e6's starts
        ("e6", "A", 12, 0, 1),
        ("e7", "A", 20, 1, 0),  # window 20-24, next to e8 (y, 25-29): no shared day
        ("e8", "A", 25, 0, 1),
    ]
    return make_events(rows)


def make_events(rows) -> pd.DataFrame:
    """Eligible dividend events from (event_id, ticker, day position, x, y) rows."""
    return pd.DataFrame([
        {"event_id": e, "ticker": t, "first_tradeable_date": DAYS[d].date(), "x": x, "y": y,
         "event_type": "dividend", "is_new_announcement": True, "extraction_status": "passed"}
        for e, t, d, x, y in rows
    ])


def test_find_clashes(prices, events):
    dropped = find_clashes(prices, events, SPECS)
    got = {(d.spec_id, d.event_id): {(r.spec_id, r.event_id) for r in d.clashes_with}
           for d in dropped}
    assert got == {
        ("x", "e1"): {("y", "e2")},
        ("x", "e3"): {("y", "e3")},
        ("x", "e5"): {("y", "e6")},
        ("y", "e2"): {("x", "e1")},
        ("y", "e3"): {("x", "e3")},
        ("y", "e6"): {("x", "e5")},
    }
    assert {d.ticker for d in dropped} == {"A"}


def test_plant_universe_drops_clashes_by_default(prices, events):
    planted, record = plant_universe("u", prices, events, SPECS, min_events=1)
    targets = {t.spec.spec_id: t.target_event_ids for t in record.truths}
    assert targets == {"x": ["e7"], "y": ["e4", "e8"]}
    assert len(record.dropped_targets) == 6
    # Nothing was planted on the dropped targets' days.
    a = planted[planted["ticker"] == "A"].set_index("date")["ret"]
    assert (a.iloc[:20] == 0).all()
    np.testing.assert_allclose(a.iloc[20:30], 0.02 / H)
    assert (a.iloc[30:] == 0).all()


def test_plant_universe_raises_on_clash(prices, events):
    with pytest.raises(ValueError, match="x:e1"):
        plant_universe("u", prices, events, SPECS, min_events=1, on_clash="raise")


def test_no_clash_with_one_spec(prices, events):
    _, record = plant_universe("u", prices, events, SPECS[:1], min_events=1)
    assert record.dropped_targets == []
    assert record.truths[0].target_event_ids == ["e1", "e3", "e5", "e7"]


def test_min_events_applies_after_drops(prices, events):
    with pytest.raises(ValueError, match="fewer than min_events"):
        plant_universe("u", prices, events, SPECS, min_events=2)


def test_dropped_targets_only_on_planted(prices, events):
    _, record = plant_universe("u", prices, events, SPECS, min_events=1)
    with pytest.raises(ValidationError):
        UniverseRecord(universe_id="p", universe_type="placebo", seed=0,
                       dropped_targets=record.dropped_targets)


def test_injected_car_within_spec_overlap(prices):
    events = make_events([
        ("e1", "A", 0, 1, 0),  # window 0-4; e2's window shares days 2-4
        ("e2", "A", 2, 1, 0),  # window 2-6
        ("e3", "A", 20, 1, 0),  # alone
        ("e4", "A", 38, 1, 0),  # window cut to 2 days by the end of the data: unmeasurable
    ])
    _, truth = make_planted(prices, events, SPECS[0], min_events=1)
    daily = 0.02 / H
    assert truth.unmeasurable_event_ids == ["e4"]
    assert truth.injected_car == pytest.approx(
        {"e1": 8 * daily, "e2": 8 * daily, "e3": 0.02, "e4": 2 * daily}
    )
    assert truth.injected_car["e1"] > SPECS[0].effect
    assert truth.injected_car_mean == pytest.approx((8 * daily * 2 + 0.02) / 3)


def test_recompute_injected_captures_cross_spec_effects(prices):
    events = make_events([("e1", "A", 0, 1, 0), ("e2", "A", 2, 0, 1)])
    short = SPECS[1].model_copy(update={"direction": -1, "effect": 0.01})
    once, tx = make_planted(prices, events, SPECS[0], min_events=1)
    twice, ty = make_planted(once, events, short, min_events=1)
    assert tx.injected_car == pytest.approx({"e1": 0.02})  # its own spec only
    tx, ty = recompute_injected(prices, twice, events, [tx, ty])
    # e1's window (0-4) gets 3 of y's days; e2's (2-6) gets 3 of x's days.
    assert tx.injected_car == pytest.approx({"e1": 0.02 - 3 * 0.01 / H})
    assert ty.injected_car == pytest.approx({"e2": -0.01 + 3 * 0.02 / H})
    assert ty.injected_car_mean == pytest.approx(ty.injected_car["e2"])


def test_plant_universe_injected_car_matches_make_planted(prices, events):
    _, record = plant_universe("u", prices, events, SPECS, min_events=1)
    for truth in record.truths:
        assert truth.injected_car == pytest.approx({e: 0.02 for e in truth.target_event_ids})


def spill_events() -> pd.DataFrame:
    events = make_events([
        ("T", "A", 2, 1, 0),  # x target, window 2-6
        ("D", "A", 6, 1, 0),  # x target clashing with Y: dropped, but shares day 6 with T
        ("Y", "A", 8, 0, 1),  # y target clashing with D: dropped
        ("Z", "B", 20, 0, 1),  # y target, window 20-24
        ("N", "A", 4, 0, 0),  # non-target, window 4-8: shares 4-6 with T
        ("F", "A", 3, 0, 0),  # would share days with T, but not eligible
        ("K", "A", 0, 0, 0),  # a buyback, window 0-4: shares 2-4 with T
        ("M", "B", 15, 0, 0),  # window 15-19, ends the day before Z's starts
        ("O", "B", 24, 0, 0),  # window 24-28: shares day 24 with Z
    ])
    events.loc[events["event_id"] == "F", "extraction_status"] = "flagged"
    events.loc[events["event_id"] == "K", "event_type"] = "buyback"
    return events


def test_spillover_in_make_planted(prices):
    events = spill_events()
    _, truth = make_planted(prices, events[events["event_id"] != "D"], SPECS[0], min_events=1)
    assert truth.spillover_event_ids == ["N", "K"]


def test_plant_universe_recomputes_spillover_on_all_events(prices):
    _, record = plant_universe("u", prices, spill_events(), SPECS, min_events=1)
    assert {d.event_id for d in record.dropped_targets} == {"D", "Y"}
    spill = {t.spec.spec_id: t.spillover_event_ids for t in record.truths}
    # D was dropped from x, so make_planted never saw it; it still gets x's effect.
    assert spill == {"x": ["D", "N", "K"], "y": ["O"]}


def test_plant_universe_min_events_defaults_to_config(prices, events):
    with pytest.raises(ValueError, match="fewer than min_events=50"):
        plant_universe("u", prices, events, SPECS)


def test_plant_universe_baseline_from_unplanted_prices(prices):
    rng = np.random.default_rng(0)
    noisy = prices.assign(ret=rng.normal(0, 0.01, len(prices)))
    benchmarks = pd.DataFrame({"ticker": "SPY", "date": DAYS, "ret": rng.normal(0, 0.01, 40)})
    events = make_events([
        ("e1", "A", 0, 1, 0), ("e2", "A", 10, 1, 0), ("e3", "B", 5, 1, 0),
        ("e4", "A", 20, 0, 1), ("e5", "B", 20, 0, 1), ("e6", "B", 30, 0, 1),
    ])
    planted, record = plant_universe("u", noisy, events, SPECS, min_events=1,
                                     benchmarks=benchmarks)
    _, plain = make_planted(noisy, events, SPECS[0], min_events=1, benchmarks=benchmarks)
    for truth in record.truths:
        targets = events[events["event_id"].isin(truth.target_event_ids)]
        before = event_cars(noisy, benchmarks, targets, H)
        assert truth.baseline_car == dict(zip(before["event_id"], before["car"], strict=True))
        assert truth.baseline_car_mean == pytest.approx(before["car"].mean())
        assert truth.baseline_car_std == pytest.approx(before["car"].std(ddof=1))
        assert truth.snr == pytest.approx(0.02 / (before["car"].std(ddof=1) / np.sqrt(3)))
        after = event_cars(planted, benchmarks, targets, H)
        np.testing.assert_allclose(after["car"], before["car"] + 0.02, atol=1e-12)
    assert record.truths[0].baseline_car == plain.baseline_car
