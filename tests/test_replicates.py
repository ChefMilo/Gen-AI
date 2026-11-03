"""Independent replicate universes (universes.replicate, DataConfig.replicates) and the
rule that variants of one base never share a universe."""

import numpy as np
import pandas as pd
import pytest

from evaluation.final_grid import GridConfig
from evaluation.oracle import oracle_table
from evaluation.plan import (
    REPO_ROOT,
    DataConfig,
    ExperimentConfig,
    _target_key,
    load_experiment_config,
    plan_experiment,
)
from evaluation.run import run_experiment
from evaluation.universe_store import entry_prices, load_universe
from experiment_helpers import small_data
from universes.replicate import resample_idiosyncratic


@pytest.fixture(scope="module")
def data():
    return small_data()


@pytest.fixture(scope="module")
def grid():
    g = load_experiment_config(REPO_ROOT / "configs" / "final.yaml").grid
    return GridConfig(strengths=g.strengths[-1:], directions=g.directions[-1:], bases=g.bases)


def config(grid, replicates: str, **overrides) -> ExperimentConfig:
    values = dict(experiment_id="rep", n_planted=8, n_placebo=2, include_real=True,
                  plant_min_events=5, base_seed=10000, systems=["null"], backtest_budget=30,
                  grid=grid, data=DataConfig(replicates=replicates))
    return ExperimentConfig(**{**values, **overrides})


def test_resample_keeps_everything_but_the_noise(data):
    prices, benchmarks = data.prices, data.benchmarks
    a = resample_idiosyncratic(prices, benchmarks, 1)
    assert a.equals(resample_idiosyncratic(prices, benchmarks, 1))  # deterministic
    b = resample_idiosyncratic(prices, benchmarks, 2)
    assert not np.allclose(a["ret"].fillna(0), b["ret"].fillna(0))
    unchanged = [c for c in prices.columns if c not in ("ret", "adj_close")]
    pd.testing.assert_frame_equal(a[unchanged], prices[unchanged])
    assert (a["ret"].isna() == prices["ret"].isna()).all()

    spy = benchmarks[benchmarks["ticker"] == "SPY"].set_index("date")["ret"]
    for _, rows in prices.groupby("ticker"):
        new = a.loc[rows.index]
        m = rows["date"].map(spy)
        ok = rows["ret"].notna() & m.notna()
        beta_old = np.polyfit(m[ok], rows["ret"][ok], 1)[0]
        beta_new = np.polyfit(m[ok], new["ret"][ok], 1)[0]
        assert beta_new == pytest.approx(beta_old, abs=0.25)
        assert new["ret"][ok].std() == pytest.approx(rows["ret"][ok].std(), rel=0.15)
        # adj_close is rebuilt from the new ret
        growth = (new["adj_close"] / new["adj_close"].shift()).iloc[1:] - 1
        assert np.allclose(growth, new["ret"].iloc[1:])


def test_planted_and_placebo_universes_get_their_own_draw(grid, data):
    plan = plan_experiment(data.prices, data.events, config(grid, "resample_noise"),
                           data.benchmarks)
    seeds = {}
    for entry in plan.universes:
        if entry.universe_type == "real":
            assert entry.noise_seed is None
        else:
            assert entry.noise_seed == entry.seed
            seeds[entry.universe_id] = entry.noise_seed
    assert len(set(seeds.values())) == len(seeds)
    placebo = plan.by_type("placebo")[0]
    assert not entry_prices(placebo, data)["ret"].equals(data.prices["ret"])
    shared = plan_experiment(data.prices, data.events, config(grid, "shared"), data.benchmarks)
    assert all(e.noise_seed is None for e in shared.universes)


def test_resample_noise_only_for_synthetic():
    with pytest.raises(ValueError, match="only for synthetic"):
        DataConfig(source="real_prices_fake_events", replicates="resample_noise")


def _t_spread(tmp_path, grid, data, replicates: str) -> pd.Series:
    """Per spec appearing more than once: the range of its oracle t-stats."""
    cfg = config(grid, replicates, n_placebo=1, include_real=False)
    out = tmp_path / replicates
    outcome = run_experiment(cfg, [], out, data=data, dry_run=True)
    table = oracle_table(outcome.plan, data, out, alpha=0.05, min_events=5)
    counts = table["spec_id"].value_counts()
    repeated = table[table["spec_id"].isin(counts[counts > 1].index)]
    assert len(repeated), "the plan should plant some spec more than once"
    return repeated.groupby("spec_id")["t"].agg(lambda t: t.max() - t.min())


def test_repeated_plants_are_independent_draws(tmp_path, grid, data):
    """Two appearances of a spec give materially different t-stats with resample_noise,
    and (near-)identical ones on shared base prices."""
    independent = _t_spread(tmp_path, grid, data, "resample_noise")
    shared = _t_spread(tmp_path, grid, data, "shared")
    # Per spec, two independent draws of t differ by |N(0, 2)|, which can be small by
    # chance; the shared draws must not differ at all.
    assert (shared < 0.01).all(), shared
    assert (independent > 0.05).all(), independent
    assert independent.mean() > 5 * max(shared.mean(), 0.05), (independent, shared)


def test_load_reproduces_a_resampled_universe(tmp_path, grid, data):
    cfg = config(grid, "resample_noise", n_planted=4, n_placebo=1, include_real=False)
    outcome = run_experiment(cfg, [], tmp_path, data=data, dry_run=True)
    for entry in outcome.plan.universes:
        load_universe(tmp_path, entry, outcome.plan.fingerprint, data)  # verifies hashes


def test_variants_of_a_base_never_share_a_universe(data):
    g = load_experiment_config(REPO_ROOT / "configs" / "final.yaml").grid
    two = GridConfig(strengths=g.strengths[-2:], directions=g.directions[-2:], bases=g.bases)
    # Even with every clash limit off, one base's strengths stay apart.
    cfg = ExperimentConfig(experiment_id="same", n_planted=8, n_placebo=0, include_real=False,
                           plant_min_events=1, max_dropped_share=1.0, specs_per_universe=4,
                           systems=["null"], base_seed=10000, grid=two)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    for entry in plan.by_type("planted"):
        keys = [_target_key(s) for s in entry.specs]
        assert len(keys) == len(set(keys)), [s.spec_id for s in entry.specs]
