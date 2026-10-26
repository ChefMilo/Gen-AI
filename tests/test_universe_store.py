"""universe_store: exact reloads from cached deltas, cache reuse and rebuilds."""

import subprocess

import pandas as pd
import pytest

from evaluation.plan import data_fingerprint, plan_experiment
from evaluation.universe_record import plant_universe
from evaluation.universe_store import (
    META,
    PRICES_DELTA,
    BaseData,
    StaleCacheError,
    apply_prices_delta,
    build_universe,
    ensure_git_ignored,
    ensure_universe,
    load_universe,
    prices_delta,
    universe_dir,
)
from experiment_helpers import config, small_data
from test_metrics_end_to_end import SPECS


@pytest.fixture(scope="module")
def data():
    return small_data()


@pytest.fixture(scope="module")
def plan(data):
    return plan_experiment(data.prices, data.events, config(), data.benchmarks)


@pytest.fixture(scope="module")
def entries(plan):
    return {t: plan.by_type(t)[0] for t in ("planted", "placebo", "real")}


@pytest.mark.parametrize("universe_type", ["planted", "placebo", "real"])
def test_cached_universe_reloads_identically(tmp_path, data, plan, entries, universe_type):
    entry = entries[universe_type]
    direct = build_universe(entry, data)
    status = ensure_universe(tmp_path, entry, plan.fingerprint, data)
    assert status.status == "built"
    loaded = load_universe(tmp_path, entry, plan.fingerprint, data)
    pd.testing.assert_frame_equal(loaded.prices, direct.prices, check_exact=True)
    pd.testing.assert_frame_equal(loaded.events, direct.events, check_exact=True)
    assert loaded.record == direct.record
    if universe_type == "planted":
        assert len(direct.record.truths) == len(entry.specs)
        assert not loaded.prices["ret"].equals(data.prices["ret"])
    if universe_type == "placebo":
        assert loaded.events["placebo"].all()


def test_planted_delta_is_small_and_replays_overlapping_specs(data):
    """Three specs with cross-spec clashes and a within-spec overlap: the ret rows alone
    reproduce adj_close exactly."""
    planted, record = plant_universe("u", data.prices, data.events, SPECS, 20)
    delta = prices_delta(data.prices, planted)
    assert 0 < len(delta) < 0.2 * len(data.prices)
    assert (planted["adj_close"] != data.prices["adj_close"]).mean() > 0.5  # not stored
    replayed = apply_prices_delta(data.prices, data.events, record, delta)
    pd.testing.assert_frame_equal(replayed, planted, check_exact=True)


def test_cache_is_reused_then_rebuilt_when_the_data_changes(tmp_path, data, plan, entries):
    entry = entries["planted"]
    first = ensure_universe(tmp_path, entry, plan.fingerprint, data)
    meta = universe_dir(tmp_path, entry.universe_id) / META
    written = meta.stat().st_mtime_ns
    second = ensure_universe(tmp_path, entry, plan.fingerprint, data)
    assert (first.status, second.status) == ("built", "reused")
    assert meta.stat().st_mtime_ns == written

    prices = data.prices.copy()
    prices.loc[prices.index[0], "volume"] += 1
    changed = BaseData(prices, data.benchmarks, data.events)
    fingerprint = data_fingerprint(changed.prices, changed.benchmarks, changed.events)
    assert fingerprint != plan.fingerprint
    with pytest.raises(StaleCacheError):
        load_universe(tmp_path, entry, fingerprint, changed)
    assert ensure_universe(tmp_path, entry, fingerprint, changed).status == "built"
    with pytest.raises(StaleCacheError):  # the old data no longer matches the cache
        load_universe(tmp_path, entry, plan.fingerprint, data)


def test_cache_is_rebuilt_when_the_entry_changes(tmp_path, data, plan, entries):
    entry = entries["placebo"]
    ensure_universe(tmp_path, entry, plan.fingerprint, data)
    other = entry.model_copy(update={"seed": entry.seed + 1})
    assert ensure_universe(tmp_path, other, plan.fingerprint, data).status == "built"


def test_corrupt_delta_is_caught_on_load(tmp_path, data, plan, entries):
    entry = entries["planted"]
    ensure_universe(tmp_path, entry, plan.fingerprint, data)
    path = universe_dir(tmp_path, entry.universe_id) / PRICES_DELTA
    delta = pd.read_parquet(path)
    delta.loc[0, "ret"] += 1e-6
    delta.to_parquet(path, index=False)
    with pytest.raises(StaleCacheError, match="prices differ"):
        load_universe(tmp_path, entry, plan.fingerprint, data)


def test_ensure_git_ignored(tmp_path):
    ensure_git_ignored(tmp_path / "out")  # not in a repository
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(RuntimeError, match="not git-ignored"):
        ensure_git_ignored(tmp_path / "results" / "x")
    (tmp_path / ".gitignore").write_text("/results/*\n", encoding="utf-8")
    ensure_git_ignored(tmp_path / "results" / "x")
