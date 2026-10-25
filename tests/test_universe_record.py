"""Clashes between targets of different specs in plant_universe."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from evaluation.universe_record import UniverseRecord, find_clashes, plant_universe
from universes.planted import PlantSpec

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
