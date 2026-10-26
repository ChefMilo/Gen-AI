"""Small fake data and configs for the experiment runner tests."""

from functools import cache

from evaluation.plan import ExperimentConfig
from evaluation.universe_store import BaseData
from universe_helpers import zero_noise
from universes.fake_events import make_fake_events
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"
MIN_EVENTS = 30  # keeps 24 specs on this data; covering them needs 12 planted universes
N_PLANTED = 12


@cache
def small_data(noise: bool = True) -> BaseData:
    """30 tickers, 2015-2020. Without noise every stock's ret equals SPY's."""
    benchmarks = make_synthetic_benchmarks(START, END, seed=0)
    prices = make_synthetic_prices(TICKERS, START, END, seed=0)
    if not noise:
        prices = zero_noise(prices, benchmarks)
    return BaseData(prices, benchmarks, make_fake_events(prices, seed=0))


def config(**overrides) -> ExperimentConfig:
    values = {
        "experiment_id": "test", "n_planted": N_PLANTED, "n_placebo": 3,
        "plant_min_events": MIN_EVENTS, "n_boot": 50, "backtest_budget": 5,
        "systems": ["null", "random"],
    }
    return ExperimentConfig(**{**values, **overrides})
