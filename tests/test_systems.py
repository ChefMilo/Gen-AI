"""UniverseView hides the universe type; stub systems."""

import dataclasses

import pandas as pd
import pytest

from evaluation.plan import plan_experiment
from evaluation.systems import (
    STUB_FINDING_SYSTEM,
    SYSTEMS,
    NullSystem,
    OracleSystem,
    RandomSystem,
    ResearchSystem,
    make_systems,
    make_view,
)
from evaluation.universe_record import UniverseRecord
from evaluation.universe_store import build_universe
from experiment_helpers import config, small_data
from universes.placebo import PlaceboInfo
from universes.planted import PlantSpec, PlantTruth

VIEW_FIELDS = {"universe_id", "prices", "benchmarks", "events", "research_start",
               "research_end"}
HIDDEN_TYPES = (PlantTruth, PlantSpec, UniverseRecord, PlaceboInfo)


@pytest.fixture(scope="module")
def universes():
    data = small_data()
    plan = plan_experiment(data.prices, data.events, config(), data.benchmarks)
    return {t: build_universe(plan.by_type(t)[0], data) for t in ("planted", "placebo", "real")}


@pytest.fixture(scope="module")
def views(universes):
    return {t: make_view(u.entry.universe_id, u.prices, u.benchmarks, u.events)
            for t, u in universes.items()}


def _reachable(obj, depth=0):
    """Every object reachable from obj through attributes, containers and frame cells."""
    yield obj
    if depth > 4:
        return
    if isinstance(obj, pd.DataFrame):
        yield from _reachable(obj.attrs, depth + 1)
        for column in obj.columns:
            if obj[column].dtype == object:
                for value in obj[column].head(200):
                    yield from _reachable(value, depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _reachable(k, depth + 1)
            yield from _reachable(v, depth + 1)
    elif isinstance(obj, list | tuple | set):
        for v in obj:
            yield from _reachable(v, depth + 1)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield from _reachable(getattr(obj, f.name), depth + 1)


def test_view_has_only_the_allowed_fields(views):
    for view in views.values():
        assert {f.name for f in dataclasses.fields(view)} == VIEW_FIELDS
        with pytest.raises(dataclasses.FrozenInstanceError):
            view.universe_id = "x"


def test_no_truth_or_placebo_field_is_reachable(views):
    for view in views.values():
        for obj in _reachable(view):
            assert not isinstance(obj, HIDDEN_TYPES)
        for frame in (view.prices, view.benchmarks, view.events):
            assert frame.attrs == {}
            for column in frame.columns:
                assert "placebo" not in column and "original" not in column
                assert "truth" not in column and "spec" not in column


def test_views_look_the_same_in_every_universe_type(views):
    planted, placebo, real = views["planted"], views["placebo"], views["real"]
    for name in ("prices", "benchmarks", "events"):
        a, b, c = (getattr(v, name) for v in (planted, placebo, real))
        assert list(a.columns) == list(b.columns) == list(c.columns)
        assert a.dtypes.equals(b.dtypes) and a.dtypes.equals(c.dtypes)
        for frame in (a, b, c):
            assert frame.index.equals(pd.RangeIndex(len(frame)))
    for view in views.values():
        assert view.events["is_new_announcement"].all()
        assert (view.events["extraction_status"] == "passed").all()
        events = view.events
        assert events.equals(events.sort_values(["first_tradeable_date", "ticker",
                                                 "event_id"]).reset_index(drop=True))


def test_view_frames_are_copies(universes, views):
    universe, view = universes["planted"], views["planted"]
    view.prices.loc[0, "ret"] = 99.0
    assert universe.prices["ret"].iloc[0] != 99.0


def test_stub_systems_follow_the_protocol(views):
    view = views["planted"]
    for system in (NullSystem(), RandomSystem(), OracleSystem()):
        assert isinstance(system, ResearchSystem)
        run = system.run(view, budget=4, seed=1)
        assert run.session_id.startswith(system.name)
    assert NullSystem().run(view, 4, 1).findings == []


def test_random_system_is_seeded(views):
    view = views["placebo"]
    a = RandomSystem().run(view, budget=6, seed=3)
    b = RandomSystem().run(view, budget=6, seed=3)
    c = RandomSystem().run(view, budget=6, seed=4)
    assert a == b and a != c
    assert len(a.findings) == 6
    assert {f.system for f in a.findings} == {STUB_FINDING_SYSTEM} == {"stub"}
    visible = set(view.events["event_id"])
    for f in a.findings:
        assert set(f.event_ids) <= visible and f.n_events == len(f.event_ids)


def test_oracle_reports_each_truth(universes, views):
    universe, view = universes["planted"], views["planted"]
    run = OracleSystem().run_with_truths(view, 5, 0, universe.record.truths)
    assert len(run.findings) == len(universe.record.truths)
    assert all(f.passed_correction for f in run.findings)
    assert all(f.system == "stub" for f in run.findings)  # never mistaken for R0's
    assert OracleSystem().run(view, 5, 0).findings == []  # no truths without the runner


def test_oracle_is_never_registered():
    assert "oracle" not in SYSTEMS
    assert OracleSystem not in SYSTEMS.values()
    with pytest.raises(ValueError, match="unknown systems"):
        make_systems(["oracle"])
    assert [s.name for s in make_systems(["null", "random"])] == ["null", "random"]

