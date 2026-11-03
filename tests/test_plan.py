"""plan_experiment: coverage, clash limits, determinism, seeds and the data fingerprint."""

import logging

import numpy as np
import pytest

from evaluation.plan import (
    PLACEBO_SEED_OFFSET,
    ClashIndex,
    ExperimentConfig,
    _Packer,
    _target_key,
    assign_specs,
    data_fingerprint,
    load_experiment_config,
    load_plan,
    plan_experiment,
    save_plan,
)
from evaluation.universe_record import find_clashes
from experiment_helpers import MIN_EVENTS, config, small_data
from universes.planted import PlantSpec


@pytest.fixture(scope="module")
def data():
    return small_data()


@pytest.fixture(scope="module")
def plan(data):
    return plan_experiment(data.prices, data.events, config(), data.benchmarks)


def keys(specs):
    return [_target_key(s) for s in specs]


def test_plan_is_deterministic_per_seed(data, plan):
    again = plan_experiment(data.prices, data.events, config(), data.benchmarks)
    assert again == plan
    other = plan_experiment(data.prices, data.events, config(base_seed=1), data.benchmarks)
    assert [u.universe_id for u in other.universes] != [u.universe_id for u in plan.universes]


def test_every_kept_spec_is_covered_evenly(plan):
    assert len(plan.kept_specs) == 24
    assert set(plan.appearances) == {s.spec_id for s in plan.kept_specs}
    assert min(plan.appearances.values()) >= 1
    assert max(plan.appearances.values()) - min(plan.appearances.values()) <= 1
    counted = {}
    for entry in plan.by_type("planted"):
        ids = [s.spec_id for s in entry.specs]
        assert len(ids) == len(set(ids)) and 1 <= len(ids) <= 4
        for spec_id in ids:
            counted[spec_id] = counted.get(spec_id, 0) + 1
    assert counted == plan.appearances


def test_planted_universes_keep_each_spec_within_the_clash_limits(data, plan):
    for entry in plan.by_type("planted"):
        index = ClashIndex(data.prices, data.events, entry.specs)
        dropped = index.dropped(keys(entry.specs))
        for key, d in zip(keys(entry.specs), dropped, strict=True):
            n = index.n_targets(key)
            assert n - d >= MIN_EVENTS
            assert d <= plan.config.max_dropped_share * n
        assert plan.planned_clashes[entry.universe_id] == sum(dropped)


def test_no_single_swap_improves_the_mix_or_clashes(data, plan):
    """Strengths and feature types mix where possible: no swap of two specs between
    universes lowers (over limits, repeated strengths, missing feature type, clashes).

    On this data buyback_large clashes with every spec but dividend_raise, so four
    universes are structured only; the plan cannot mix more than it does.
    """
    planted = [list(u.specs) for u in plan.by_type("planted")]
    index = ClashIndex(data.prices, data.events, plan.kept_specs)
    packer = _Packer(plan.kept_specs, index, len(planted), 4, MIN_EVENTS, 0.1, 0)
    costs = [packer.cost(u) for u in planted]
    for a in range(len(planted)):
        for b in range(a + 1, len(planted)):
            for i, x in enumerate(planted[a]):
                for j, y in enumerate(planted[b]):
                    ids_a = {s.spec_id for s in planted[a]}
                    ids_b = {s.spec_id for s in planted[b]}
                    if y.spec_id in ids_a or x.spec_id in ids_b:
                        continue
                    new_a = planted[a][:i] + [y] + planted[a][i + 1:]
                    new_b = planted[b][:j] + [x] + planted[b][j + 1:]
                    swapped = [p + q for p, q in zip(packer.cost(new_a), packer.cost(new_b),
                                                     strict=True)]
                    current = [p + q for p, q in zip(costs[a], costs[b], strict=True)]
                    assert swapped >= current
    mixed = sum(len({s.feature_type for s in u}) == 2 for u in planted)
    assert mixed == 4


def test_clash_index_counts_equal_find_clashes(data, plan):
    for entry in plan.by_type("planted")[:3]:
        specs = entry.specs + [plan.kept_specs[0]]  # add one that clashes
        specs = list({s.spec_id: s for s in specs}.values())
        index = ClashIndex(data.prices, data.events, specs)
        assert sum(index.dropped(keys(specs))) == len(find_clashes(data.prices, data.events,
                                                                   specs))


def test_clashes_no_worse_than_random_assignments(data, plan):
    planted = plan.by_type("planted")
    sizes = [len(u.specs) for u in planted]
    slots = [s for u in planted for s in u.specs]
    index = ClashIndex(data.prices, data.events, plan.kept_specs)
    rng = np.random.default_rng(0)
    planned = sum(plan.planned_clashes.values())
    for _ in range(20):
        order = rng.permutation(len(slots))
        shuffled, total, k = [slots[i] for i in order], 0, 0
        for size in sizes:
            total += sum(index.dropped(keys(shuffled[k:k + size])))
            k += size
        assert planned <= total


def test_placebo_and_planted_seeds(plan):
    placebo = [u.seed for u in plan.by_type("placebo")]
    assert placebo == [PLACEBO_SEED_OFFSET + i for i in range(3)]
    planted = [u.seed for u in plan.by_type("planted")]
    assert planted == list(range(12))
    assert len(set(placebo + planted)) == len(placebo) + len(planted)
    for entry in plan.by_type("planted"):
        assert {s.seed for s in entry.specs} == {entry.seed}
    assert [u.seed for u in plan.by_type("real")] == [None]


def test_universe_ids_are_opaque(plan):
    ids = [u.universe_id for u in plan.universes]
    assert len(set(ids)) == len(ids)
    for universe_id in ids:
        assert not any(word in universe_id for word in ("planted", "placebo", "real"))


def test_too_few_planted_universes_raise(data):
    with pytest.raises(ValueError, match="raise n_planted"):
        plan_experiment(data.prices, data.events, config(n_planted=11))
    with pytest.raises(ValueError, match="do not fit"):
        plan_experiment(data.prices, data.events, config(n_planted=2))


def test_fingerprint_changes_with_the_data(data):
    base = data_fingerprint(data.prices, data.benchmarks, data.events)
    assert data_fingerprint(data.prices, data.benchmarks, data.events) == base

    prices = data.prices.copy()
    prices.loc[prices.index[100], "ret"] += 1e-9
    assert data_fingerprint(prices, data.benchmarks, data.events).prices_hash != base.prices_hash

    fewer = data.events.iloc[1:]
    changed = data_fingerprint(data.prices, data.benchmarks, fewer)
    assert changed.event_ids_hash != base.event_ids_hash and changed.n_events == base.n_events - 1

    events = data.events.copy()
    events.loc[events.index[0], "dividend_per_share"] = 9.99
    edited = data_fingerprint(data.prices, data.benchmarks, events)
    assert edited.event_ids_hash == base.event_ids_hash
    assert edited.events_hash != base.events_hash


def test_plan_round_trips_through_json(plan, tmp_path):
    save_plan(plan, tmp_path / "plan.json")
    assert load_plan(tmp_path / "plan.json") == plan


def test_experiment_config_defaults():
    cfg = load_experiment_config()
    assert cfg.systems == ["null", "random"]
    resolved = ExperimentConfig(experiment_id="x").resolved()
    assert resolved.plant_min_events == 50 and resolved.min_gap_days == 10
    assert resolved.backtest_budget == 30 and resolved.jaccard_min == 0.6
    assert resolved.research_start is not None


# --- Balanced appearances ----------------------------------------------------------


class StubIndex:
    """A ClashIndex stand-in: 100 targets per base, all dropped when the universe holds
    a base it clashes with."""

    def __init__(self, specs, clashes: set[frozenset]):
        self.keys = [(s.spec_id.split("_")[0],) for s in specs]
        self.clashes = clashes

    def n_targets(self, key) -> int:
        return 100

    def dropped(self, keys) -> list[int]:
        return [100 if any(frozenset({a[0], b[0]}) in self.clashes
                           for j, b in enumerate(keys) if j != i) else 0
                for i, a in enumerate(keys)]


STRENGTHS = ("weak", "moderate", "strong", "very_strong")


def variants(base: str, feature_type: str = "structured") -> list[PlantSpec]:
    return [PlantSpec(spec_id=f"{base}_{k}", event_type="dividend", effect=0.01,
                      strength=strength, complexity="single", feature_type=feature_type)
            for k, strength in enumerate(STRENGTHS)]


def appearances(universes) -> dict[str, int]:
    out = {}
    for members in universes:
        for m in members:
            out[m.spec_id] = out.get(m.spec_id, 0) + 1
    return out


def test_balanced_appearances_when_an_assignment_exists():
    """The final grid's shape: base a clashes with b and c, d fits with all. Cover + fill
    leaves a short (each a-variant once); {a, d} x 8 + {b, c} x 8 gives 2 each."""
    specs = (variants("a", "text") + variants("b") + variants("c")
             + variants("d", "text"))
    index = StubIndex(specs, {frozenset("ab"), frozenset("ac")})
    greedy = _Packer(specs, index, 16, 4, 50, 0.1, 0)
    universes = greedy.cover()
    greedy.fill(universes)
    assert min(appearances(universes).values()) == 1  # the case the search is for

    for seed in (0, 1, 2):
        assigned = assign_specs(specs, index, 16, 4, 50, seed, max_dropped_share=0.1)
        assert set(appearances(assigned).values()) == {2}
        assert all(assigned)  # no planted universe left empty
        for members in assigned:
            bases = {m.spec_id[0] for m in members}
            assert len(bases) == len(members) and not ({"a", "b"} <= bases
                                                       or {"a", "c"} <= bases)
        again = assign_specs(specs, index, 16, 4, 50, seed, max_dropped_share=0.1)
        assert again == assigned  # deterministic per seed


def test_uneven_appearances_fall_back_and_are_logged(caplog):
    """Three universes, each holding at most one variant of x: x cannot get a second
    appearance, so the fill result stays and its short specs are logged."""
    specs = variants("x")[:3] + variants("y")[:1]
    index = StubIndex(specs, set())
    with caplog.at_level(logging.WARNING, logger="evaluation.plan"):
        assigned = assign_specs(specs, index, 3, 2, 50, 0, max_dropped_share=0.1)
    counts = appearances(assigned)
    assert counts == {"x_0": 1, "x_1": 1, "x_2": 1, "y_0": 2}
    assert "planted appearances uneven (1 to 2)" in caplog.text
    assert "3 specs with 1: x_0, x_1, x_2" in caplog.text
