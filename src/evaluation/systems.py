"""The research-system interface the experiment runner drives, and stub systems.

A system sees one universe through a UniverseView: its prices, benchmarks, eligible
events, universe_id and research period. The view carries nothing that tells the
universe type: no truths, no plan or spec list, no seed, no placebo columns
(PLACEBO_COLUMNS) and no original dates. Universe ids are opaque
(evaluation.plan). Events are sorted and re-indexed the same way in every universe,
and frames are copies with empty attrs.

ResearchSystem.run(view, budget, seed) -> SystemRun. Real systems (R0-R3) will
implement it; until then:
- NullSystem: no findings;
- RandomSystem: random event subsets, reference CAR as the effect, significant at a
  random rate. On a placebo universe its significant findings are false discoveries;
- OracleSystem (tests only, never registered): the runner hands it the universe's
  truths separately (run_with_truths); it reports each truth's measurable targets.

Stub findings carry Finding.system = STUB_FINDING_SYSTEM ("stub"), so they are never
mistaken for R0's; the runner keys results by ResearchSystem.name, not by that field.

Registry: SYSTEMS maps the names the CLI accepts to factories. Code outside
src/evaluation adds its systems with register_system(name, factory) at import time, in a
module listed in the experiment config:

    systems:
      names: [null, r0, r2]
      modules: [agent.systems]  # imported before the names are looked up

resolve_system_names imports those modules (only when a requested name is not already
registered) and checks the names. agent.experiment,
which updates SYSTEMS directly, is still tried for names nothing else registered, with a
deprecation warning.
"""

import importlib
import inspect
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats as sp_stats

from agent.finding import EvidenceQuote, Finding
from baselines.r0 import R0System
from dsl.signal import SignalSpec
from evaluation.reference_car import event_cars
from evaluation.universe_record import measurable_target_ids
from universes.placebo import PLACEBO_COLUMNS
from universes.planted import PlantTruth, eligible

log = logging.getLogger(__name__)

STUB_FINDING_SYSTEM = "stub"
VIEW_EVENT_ORDER = ["first_tradeable_date", "ticker", "event_id"]
METADATA_FIELDS = [
    "backtests_used",
    "tool_calls",
    "tokens",
    "cost_usd",
    "wall_time_s",
    "trace_path",
]


@dataclass(frozen=True)
class UniverseView:
    """What a research system may see of a universe."""

    universe_id: str
    prices: pd.DataFrame
    benchmarks: pd.DataFrame
    events: pd.DataFrame  # eligible events only
    research_start: date | None
    research_end: date | None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.attrs = {}
    return out


def make_view(
    universe_id: str,
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    events: pd.DataFrame,
    research_start: date | None = None,
    research_end: date | None = None,
) -> UniverseView:
    """The view of a universe's frames: eligible events without placebo columns."""
    shown = eligible(events).drop(columns=PLACEBO_COLUMNS, errors="ignore")
    shown = shown.sort_values(VIEW_EVENT_ORDER, kind="stable")
    return UniverseView(
        universe_id=universe_id,
        prices=_clean(prices),
        benchmarks=_clean(benchmarks),
        events=_clean(shown),
        research_start=research_start,
        research_end=research_end,
    )


class SystemRun(BaseModel):
    """One research session's output.

    metadata keys (all optional): backtests_used, tool_calls, tokens, cost_usd,
    wall_time_s, trace_path. The runner fills wall_time_s when it is missing.
    """

    findings: list[Finding]
    session_id: str
    metadata: dict[str, Any] = {}


@runtime_checkable
class ResearchSystem(Protocol):
    name: str

    def run(self, view: UniverseView, budget: int, seed: int) -> SystemRun: ...


def session_id(system: str, view: UniverseView, seed: int) -> str:
    return f"{system}-{view.universe_id}-{seed}"


def _finding(
    sid: str,
    k: int,
    event_type: str,
    event_ids: list[str],
    cars: pd.DataFrame,
    passed: bool,
    rationale: str,
    n_tests: int,
) -> Finding:
    """A Finding over event_ids with the reference CAR stats of cars; the corrected
    p-value is Bonferroni over n_tests (passed_correction is set by the caller)."""
    car = cars["car"].astype(float)
    n = len(car)
    mean = float(car.mean()) if n else 0.0
    sd = float(car.std(ddof=1)) if n > 1 else float("nan")
    t_stat = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else 0.0
    p_raw = float(2 * sp_stats.norm.sf(abs(t_stat)))
    return Finding(
        finding_id=f"{sid}-f{k}",
        session_id=sid,
        system=STUB_FINDING_SYSTEM,
        signal_id=f"{sid}-s{k}",
        signal_spec=SignalSpec(event_type=event_type),
        event_type=event_type,
        event_ids=event_ids,
        n_events=len(event_ids),
        direction="long" if mean >= 0 else "short",
        effect_car=mean,
        t_stat=t_stat,
        p_value_raw=p_raw,
        p_value_corrected=min(1.0, p_raw * max(n_tests, 1)),
        passed_correction=passed,
        rationale=rationale,
        evidence=[EvidenceQuote(event_id=event_ids[0], quote="stub system: no quote")],
    )


class NullSystem:
    """Finds nothing."""

    name = "null"

    def run(self, view: UniverseView, budget: int, seed: int) -> SystemRun:
        return SystemRun(
            findings=[],
            session_id=session_id(self.name, view, seed),
            metadata={"backtests_used": 0, "tool_calls": 0, "tokens": 0, "cost_usd": 0.0},
        )


class RandomSystem:
    """budget random backtests: a random event type and subset of its events each.

    Each backtest is reported as a finding; passed_correction is drawn with
    probability significant_rate.
    """

    name = "random"

    def __init__(
        self,
        significant_rate: float = 0.2,
        min_events: int = 20,
        max_events: int = 200,
        holding_days: int = 5,
    ):
        self.significant_rate = significant_rate
        self.min_events, self.max_events = min_events, max_events
        self.holding_days = holding_days

    def run(self, view: UniverseView, budget: int, seed: int) -> SystemRun:
        rng = np.random.default_rng(seed)
        sid = session_id(self.name, view, seed)
        by_type = {
            t: g["event_id"].tolist() for t, g in view.events.groupby("event_type", sort=True)
        }
        by_type = {t: ids for t, ids in by_type.items() if len(ids) >= self.min_events}
        findings = []
        for k in range(budget if by_type else 0):
            event_type = sorted(by_type)[int(rng.integers(len(by_type)))]
            pool = by_type[event_type]
            size = int(rng.integers(self.min_events, min(self.max_events, len(pool)) + 1))
            ids = sorted(rng.choice(pool, size=size, replace=False).tolist())
            cars = event_cars(
                view.prices,
                view.benchmarks,
                view.events[view.events["event_id"].isin(ids)],
                self.holding_days,
            )
            if cars.empty:
                continue
            passed = bool(rng.random() < self.significant_rate)
            findings.append(
                _finding(sid, k, event_type, ids, cars, passed, "random subset", budget)
            )
        return SystemRun(
            findings=findings,
            session_id=sid,
            metadata={
                "backtests_used": budget if by_type else 0,
                "tool_calls": budget if by_type else 0,
                "tokens": 0,
                "cost_usd": 0.0,
            },
        )


class OracleSystem:
    """Test-only upper bound: reports every planted truth exactly. Never registered.

    The runner calls run_with_truths with the universe's truths (none on placebo and
    real universes); run() alone has no truths and finds nothing.
    """

    name = "oracle"
    needs_truths = True

    def __init__(self, holding_days: int = 5):
        self.holding_days = holding_days

    def run(self, view: UniverseView, budget: int, seed: int) -> SystemRun:
        return self.run_with_truths(view, budget, seed, [])

    def run_with_truths(
        self, view: UniverseView, budget: int, seed: int, truths: list[PlantTruth]
    ) -> SystemRun:
        sid = session_id(self.name, view, seed)
        visible = set(view.events["event_id"])
        findings = []
        for k, truth in enumerate(truths):
            ids = sorted(measurable_target_ids(truth) & visible)
            if not ids:
                continue
            cars = event_cars(
                view.prices,
                view.benchmarks,
                view.events[view.events["event_id"].isin(ids)],
                self.holding_days,
            )
            findings.append(
                _finding(
                    sid,
                    k,
                    truth.spec.event_type,
                    ids,
                    cars,
                    True,
                    f"oracle: {truth.spec.spec_id}",
                    len(truths),
                )
            )
        return SystemRun(
            findings=findings,
            session_id=sid,
            metadata={
                "backtests_used": len(findings),
                "tool_calls": 0,
                "tokens": 0,
                "cost_usd": 0.0,
            },
        )


# Systems the CLI can name: name -> factory (usually the class). OracleSystem is
# deliberately absent. Other packages add theirs with register_system, from a module
# listed under systems.modules in the experiment config (load_system_modules).
SYSTEMS: dict[str, Callable[..., ResearchSystem]] = {
    "null": NullSystem,
    "random": RandomSystem,
    "r0": R0System,
}
# name -> the module that registered it, where known.
SYSTEM_SOURCES: dict[str, str] = {name: __name__ for name in SYSTEMS}

# Deprecated: the agent's own entry point registers its systems by updating SYSTEMS.
# Tried only for names no configured module registered.
LEGACY_SYSTEM_MODULE = "agent.experiment"
LEGACY_REGISTER = "register_agent_systems"

# Options make_systems passes to a factory that accepts them.
FACTORY_OPTIONS = ("holding_days", "min_events")


class SystemConfigError(ValueError):
    """A system module failed to import, or a requested system is not registered."""


def register_system(
    name: str, factory: Callable[..., ResearchSystem], source: str | None = None
) -> None:
    """Make name available to make_systems. Registering the same factory again is a
    no-op (modules may be imported more than once); a different one raises."""
    current = SYSTEMS.get(name)
    if current is not None and current is not factory:
        raise ValueError(
            f"system {name!r} is already registered by {SYSTEM_SOURCES.get(name, '?')}"
        )
    SYSTEMS[name] = factory
    SYSTEM_SOURCES[name] = source or getattr(factory, "__module__", None) or "?"


def load_system_modules(modules: Iterable[str]) -> None:
    """Import each module, which registers its systems as a side effect."""
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as err:
            raise SystemConfigError(
                f"could not import system module {module!r} (systems.modules): "
                f"{type(err).__name__}: {err}"
            ) from err


def _legacy_fallback(names: Iterable[str]) -> None:
    """Let agent.experiment register its systems the old way, if it exists."""
    missing = [n for n in names if n not in SYSTEMS]
    if not missing:
        return
    try:
        legacy = importlib.import_module(LEGACY_SYSTEM_MODULE)
    except ModuleNotFoundError as err:
        if err.name == LEGACY_SYSTEM_MODULE or LEGACY_SYSTEM_MODULE.startswith(f"{err.name}."):
            return  # no agent.experiment here
        raise SystemConfigError(
            f"could not import fallback module {LEGACY_SYSTEM_MODULE!r}: {err}"
        ) from err
    register = getattr(legacy, LEGACY_REGISTER, None)
    if register is None:
        return
    before = set(SYSTEMS)
    register()
    for name in set(SYSTEMS) - before:
        SYSTEM_SOURCES[name] = LEGACY_SYSTEM_MODULE


def resolve_system_names(names: list[str], modules: Iterable[str] = ()) -> None:
    """Check every name is registered, importing the configured modules first if any
    name is missing (so built-in names, e.g. --systems r0, work without them).

    Names still missing are looked up in agent.experiment (deprecated), with a warning
    for each system it supplies. Raises SystemConfigError naming the failing module, or
    the unknown names and the registered ones.
    """
    if any(name not in SYSTEMS for name in names):
        load_system_modules(modules)
    _legacy_fallback(names)
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown:
        raise SystemConfigError(
            f"unknown systems {unknown}; registered: {sorted(SYSTEMS)}. "
            "List the module that registers them under systems.modules in the config."
        )
    for name in names:
        if SYSTEM_SOURCES.get(name) == LEGACY_SYSTEM_MODULE:
            log.warning(
                "DEPRECATED: system %r comes from %s; register it with "
                "evaluation.systems.register_system in a module listed under "
                "systems.modules in the experiment config instead",
                name, LEGACY_SYSTEM_MODULE,
            )


def _factory_kwargs(factory: Callable, options: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(options)
    accepted = {p.name for p in params}
    return {k: v for k, v in options.items() if k in accepted}


def make_systems(
    names: list[str],
    holding_days: int = 5,
    min_events: int = 20,
) -> list[ResearchSystem]:
    """One system per name; each factory gets the FACTORY_OPTIONS it accepts."""
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown:
        raise SystemConfigError(f"unknown systems {unknown}; registered: {sorted(SYSTEMS)}")
    options = {"holding_days": holding_days, "min_events": min_events}
    return [SYSTEMS[name](**_factory_kwargs(SYSTEMS[name], options)) for name in names]
