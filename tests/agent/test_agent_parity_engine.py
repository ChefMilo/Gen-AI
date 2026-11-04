"""Three-way parity: the interim backtester (agent.interim) == Member 4's engine
(backtest.engine.run_event_study, with the harness's minimum-events rule) == R0
(baselines.r0), on the same signals and the same synthetic universe.

Compared: n_events, event_ids, raw mean CAR, t, raw p and the test status. R0 has no
direction or general conditions, so its side is built from R0's own primitives
(select_dividend_increases / calculate_other_returns, event_cars, summarize_returns).
A fourth case uses "!=" on a field with missing values; R0 has no "!=", so it is interim
vs engine only, with a unit test of missing values under every operator.
"""

from datetime import date

import pandas as pd
import pytest

from agent.interim.ports import build_ports
from backtest.engine import run_event_study
from baselines.r0 import R0System
from dsl.signal import SignalSpec
from evaluation.reference_car import event_cars
from evaluation.systems import make_view
from universes.fake_events import make_fake_events
from universes.planted import PlantSpec, make_planted
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TOL = 1e-9
MIN_EVENTS = 20
SIGNALS = {
    "broad": {"event_type": "dividend"},
    "conditions": {"event_type": "dividend", "conditions": [
        {"field": "dividend_change_pct", "op": ">", "value": 10.0},
        {"field": "mcap_bucket", "op": "==", "value": "small"}]},
    "short": {"event_type": "buyback", "direction": "short", "conditions": [
        {"field": "is_new_program", "op": "==", "value": True}]},
}


@pytest.fixture(scope="module")
def view():
    start, end, seed = date(2015, 1, 1), date(2020, 12, 31), 9811
    tickers = [f"T{i:03d}" for i in range(40)]
    prices = make_synthetic_prices(tickers, start, end, seed=seed)
    benchmarks = make_synthetic_benchmarks(start, end, seed=seed)
    events = make_fake_events(prices, seed=seed)
    spec = PlantSpec(spec_id="parity_raise_small", event_type="dividend",
                     condition='dividend_change_pct > 10 and mcap_bucket == "small"',
                     effect=0.02, strength="strong", complexity="conjunction",
                     feature_type="structured")
    planted, _ = make_planted(prices, events, spec, min_events=20, benchmarks=benchmarks)
    return make_view("parity", planted, benchmarks, events, date(2016, 1, 1),
                     date(2020, 6, 30))


def until_end(view):
    end = pd.Timestamp(view.research_end)
    return (view.prices[pd.to_datetime(view.prices["date"]) <= end],
            view.benchmarks[pd.to_datetime(view.benchmarks["date"]) <= end])


def r0_side(view, name):
    """R0's selection for each signal, then R0's CAR filter and statistics."""
    r0 = R0System(holding_days=5, min_events=MIN_EVENTS)
    events = view.events[(view.events["first_tradeable_date"] >= view.research_start)
                         & (view.events["first_tradeable_date"] <= view.research_end)]
    if name == "broad":
        selected = events[events["event_type"].eq("dividend")]
    elif name == "conditions":
        selected = r0.select_dividend_increases(events)
        selected = selected[selected["mcap_bucket"].eq("small")]
    else:
        selected = events[events["event_type"].eq("buyback")
                          & events["is_new_program"].eq(True)]
    prices, benchmarks = until_end(view)
    cars = event_cars(prices, benchmarks, selected, r0.holding_days)
    cars = cars.loc[cars["car"].notna() & ~cars["car"].isin([float("inf"), float("-inf")])]
    return {"event_ids": cars["event_id"].tolist(), **r0.summarize_returns(cars)}


def engine_side(view, name):
    """Member 4's engine on the research-period data, as the harness serves it."""
    events = view.events[(view.events["first_tradeable_date"] >= view.research_start)
                         & (view.events["first_tradeable_date"] <= view.research_end)]
    prices, benchmarks = until_end(view)
    result = run_event_study(prices, benchmarks, events, SignalSpec(**SIGNALS[name]), name,
                             min_events=MIN_EVENTS)
    return {"event_ids": result.event_ids, "n_events": result.n_events,
            "mean_car": result.mean_car, "t_stat": result.t_stat,
            "p_value_raw": result.p_value_raw, "status": result.status}


@pytest.mark.parametrize("name", list(SIGNALS))
def test_interim_equals_engine_equals_r0(view, name, registry):
    ports = build_ports(view, budget=10, holding_days=5, min_events=MIN_EVENTS,
                        with_retriever=False, mask_dates=True)
    mine = ports.backtester.run(ports.backtester.define(SIGNALS[name]))
    engine = engine_side(view, name)
    r0 = r0_side(view, name)
    assert mine["status"] == engine["status"] == r0["status"] == "tested"
    assert mine["event_ids"] == engine["event_ids"] == r0["event_ids"]
    assert mine["n_events"] == engine["n_events"] == r0["n_events"] >= MIN_EVENTS
    for key in ("mean_car", "t_stat", "p_value_raw"):
        assert mine[key] == pytest.approx(engine[key], abs=TOL, rel=0)
        assert mine[key] == pytest.approx(r0[key], abs=TOL, rel=0)


def test_the_signals_are_not_trivial(view, registry):
    """Broad, conditional and short signals select different, non-empty event sets."""
    sets = [set(engine_side(view, name)["event_ids"]) for name in SIGNALS]
    assert all(sets) and len({frozenset(s) for s in sets}) == 3


# --- "!=" and missing values -------------------------------------------------------------
# R0 has no "!=" (its comparisons are > and ==), so this case is interim vs engine.

NOT_EQUAL = {"event_type": "dividend", "conditions": [
    {"field": "dividend_change_pct", "op": "!=", "value": 0.0}]}


def test_not_equal_on_a_field_with_missing_values_matches_the_engine(view, registry):
    ports = build_ports(view, budget=10, holding_days=5, min_events=MIN_EVENTS,
                        with_retriever=False, mask_dates=True)
    dividends = ports.events.events[ports.events.events["event_type"] == "dividend"]
    assert dividends["dividend_change_pct"].isna().any()  # first dividends have no change
    mine = ports.backtester.run(ports.backtester.define(NOT_EQUAL))
    events = view.events[(view.events["first_tradeable_date"] >= view.research_start)
                         & (view.events["first_tradeable_date"] <= view.research_end)]
    prices, benchmarks = until_end(view)
    engine = run_event_study(prices, benchmarks, events, SignalSpec(**NOT_EQUAL), "ne",
                             min_events=MIN_EVENTS)
    assert mine["status"] == engine.status == "tested"
    assert mine["event_ids"] == engine.event_ids
    assert mine["n_selected"] == engine.n_selected
    missing = set(dividends.loc[dividends["dividend_change_pct"].isna(), "event_id"])
    assert not missing & set(mine["event_ids"])
    for key, value in (("mean_car", engine.mean_car), ("t_stat", engine.t_stat),
                       ("p_value_raw", engine.p_value_raw)):
        assert mine[key] == pytest.approx(value, abs=TOL, rel=0)


OPERATORS = [("==", 5.0), ("!=", 5.0), (">", 1.0), (">=", 5.0), ("<", 20.0), ("<=", 5.0),
             ("in", [5.0, 15.0])]


@pytest.mark.parametrize("op,value", OPERATORS)
def test_missing_and_non_finite_values_never_match(op, value):
    """Every operator, interim select vs the engine's select_events on a tiny frame with a
    missing and an infinite value."""
    from agent.fakes import FakeSignalSpec
    from agent.interim.spec import select
    from backtest.selection import select_events

    events = pd.DataFrame({
        "event_id": ["nan", "inf", "five", "fifteen"],
        "event_type": ["dividend"] * 4,
        "is_new_announcement": [True] * 4,
        "extraction_status": ["passed"] * 4,
        "dividend_change_pct": [float("nan"), float("inf"), 5.0, 15.0],
    })
    spec = {"event_type": "dividend",
            "conditions": [{"field": "dividend_change_pct", "op": op, "value": value}]}
    mine = set(select(events, FakeSignalSpec.model_validate(spec))["event_id"])
    assert not mine & {"nan", "inf"}
    if op != "in":  # the engine's DSL has no "in"
        assert mine == set(select_events(events, SignalSpec(**spec))["event_id"])
