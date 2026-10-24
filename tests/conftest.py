import pytest

from universes.fake_events import make_fake_events
from universes.synthetic import make_synthetic_prices

# The documented fake-events scale (PR #2): 180 tickers, 2015-2026, about 10k events.
LARGE_TICKERS = [f"G{i:03d}" for i in range(180)]
LARGE_START, LARGE_END = "2015-01-01", "2026-12-31"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes several seconds; deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def large_fake_events():
    prices = make_synthetic_prices(LARGE_TICKERS, LARGE_START, LARGE_END, seed=0)
    return make_fake_events(prices, seed=0)
