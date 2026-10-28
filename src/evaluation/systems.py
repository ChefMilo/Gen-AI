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

Finding.system only allows R0-R3 / R3-U (agent/finding.py), so stub findings carry
STUB_FINDING_SYSTEM; the runner keys results by ResearchSystem.name, not by that field.
"""

import math
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

STUB_FINDING_SYSTEM = "R0"
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


# Systems the CLI can name. OracleSystem is deliberately absent.
SYSTEMS: dict[str, type] = {
    "null": NullSystem,
    "random": RandomSystem,
    "r0": R0System,
}


def make_systems(
    names: list[str],
    holding_days: int = 5,
    min_events: int = 20,
) -> list[ResearchSystem]:
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown systems {unknown}; known: {sorted(SYSTEMS)}")

    out = []

    for name in names:
        cls = SYSTEMS[name]

        if cls in (RandomSystem, R0System):
            system = cls(
                holding_days=holding_days,
                min_events=min_events,
            )
        else:
            system = cls()

        out.append(system)

    return out
