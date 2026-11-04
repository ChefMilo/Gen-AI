"""data.real: store: the real universe on the extracted events and real prices, while the
planted and placebo universes keep the base data."""

import pandas as pd
import pytest

from evaluation import run as runner
from evaluation.plan import DataConfig, data_fingerprint
from evaluation.run import load_real_data, run_experiment, with_real_data
from evaluation.systems import NullSystem, SystemRun
from evaluation.universe_store import BaseData
from experiment_helpers import config, small_data
from universes.fake_events import make_fake_events
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices


def other_data() -> BaseData:
    """Stands in for the store: other tickers, other events."""
    prices = make_synthetic_prices([f"R{i:02d}" for i in range(5)], "2015-01-01",
                                   "2020-12-31", seed=7)
    benchmarks = make_synthetic_benchmarks("2015-01-01", "2020-12-31", seed=7)
    return BaseData(prices, benchmarks, make_fake_events(prices, seed=7))


class Recorder:
    """Finds nothing; remembers the tickers each universe showed it."""

    name = "recorder"

    def __init__(self):
        self.tickers = {}

    def run(self, view, budget, seed):
        self.tickers[view.universe_id] = set(view.events["ticker"])
        return SystemRun(findings=[], session_id=view.universe_id)


def small(**overrides):
    return config(n_planted=0, n_placebo=2, systems=["null"], **overrides)


def test_only_the_real_universe_uses_the_real_data(tmp_path, monkeypatch):
    real = other_data()
    monkeypatch.setattr(runner, "load_real_data", lambda cfg: real)
    recorder = Recorder()
    cfg = small(data=DataConfig(real="store"))
    outcome = run_experiment(cfg, [recorder], tmp_path, data=small_data())
    real_id = outcome.plan.by_type("real")[0].universe_id
    assert recorder.tickers[real_id] <= set(real.events["ticker"])
    for entry in outcome.plan.universes:
        if entry.universe_id != real_id:
            assert recorder.tickers[entry.universe_id] <= set(small_data().events["ticker"])


def test_plan_does_not_need_the_real_data(tmp_path, monkeypatch):
    """The plan (and its frozen hash) is built from the base data alone."""
    monkeypatch.setattr(runner, "load_real_data", lambda cfg: other_data())
    with_store = run_experiment(small(data=DataConfig(real="store")), [], tmp_path / "a",
                                data=small_data(), dry_run=True)
    base = run_experiment(small(), [], tmp_path / "b", data=small_data(), dry_run=True)
    assert with_store.plan.fingerprint == base.plan.fingerprint
    assert with_store.plan.universes == base.plan.universes


def test_real_cache_follows_the_real_data(tmp_path, monkeypatch):
    real = other_data()
    monkeypatch.setattr(runner, "load_real_data", lambda cfg: real)
    cfg = small(data=DataConfig(real="store"))
    run_experiment(cfg, [NullSystem()], tmp_path, data=small_data())
    fewer = BaseData(real.prices, real.benchmarks, real.events.iloc[1:].reset_index(drop=True))
    monkeypatch.setattr(runner, "load_real_data", lambda cfg: fewer)
    outcome = run_experiment(cfg, [], tmp_path, data=small_data(), dry_run=True)
    status = {b.universe_type: b.status for b in outcome.builds}
    assert status == {"placebo": "reused", "real": "built"}


def test_base_by_default_needs_no_store(monkeypatch):
    def fail(cfg):
        raise AssertionError("loaded the store")

    monkeypatch.setattr(runner, "load_real_data", fail)
    assert with_real_data(small_data(), small()) is small_data()
    no_real = small(include_real=False, data=DataConfig(real="store"))
    assert with_real_data(small_data(), no_real) is small_data()


@pytest.mark.parametrize(("period", "first"), [("research", "2016-01-01"),
                                               ("holdout", "2023-01-01")])
def test_load_real_data_periods(monkeypatch, period, first):
    prices = make_synthetic_prices(["A", "B", "C"], "2015-01-01", "2024-12-31", seed=1)
    benchmarks = make_synthetic_benchmarks("2015-01-01", "2024-12-31", seed=1)
    events = make_fake_events(prices[prices["ticker"] != "C"], seed=1)
    seen = {}

    def fake_load_events(start=None, end=None):
        seen["range"] = (start, end)
        day = pd.to_datetime(events["first_tradeable_date"])
        keep = pd.Series(True, index=events.index)
        if start is not None:
            keep &= day >= pd.Timestamp(start)
        if end is not None:
            keep &= day <= pd.Timestamp(end)
        return events[keep].reset_index(drop=True)

    monkeypatch.setattr("store.events.load_events", fake_load_events)
    monkeypatch.setattr("store.prices.load_prices", lambda usable_only=True: prices)
    monkeypatch.setattr("store.prices.load_benchmarks", lambda: benchmarks)
    data = load_real_data(small(data=DataConfig(real="store", real_period=period)))
    assert str(seen["range"][0]) == first
    assert set(data.prices["ticker"]) == {"A", "B"}  # tickers with events only
    day = pd.to_datetime(data.events["first_tradeable_date"])
    assert day.min() >= pd.Timestamp(first)
    if period == "research":
        assert day.max() <= pd.Timestamp("2022-12-31")
        assert pd.to_datetime(data.prices["date"]).max() <= pd.Timestamp("2022-12-31")
    assert data_fingerprint(data.prices, data.benchmarks, data.events).n_events == len(data.events)
