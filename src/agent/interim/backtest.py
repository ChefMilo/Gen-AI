"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

InterimBacktester: the Backtester port over a universe view, with the testing budget.

It mirrors R0 (src/baselines/r0.py, PR #17) so the agents and R0 are compared on equal
terms:
- CAR: evaluation.reference_car.event_cars over the view's prices and SPY, day 0 =
  first_tradeable_date, holding_days trading days (one option: the runner's, as R0).
- Unmeasurable events are dropped, as R0 does: event_cars skips windows with no day-0
  row, a bad price day, no benchmark, or past the last price row. Prices end at
  research_end (the runner cuts them there; this port cuts again), so a window that runs
  past research_end is skipped as "past_data" and is not in event_ids or n_events.
- Statistics: R0's summarize_returns, i.e. scipy.stats.ttest_1samp against 0 on the
  finite CARs. status "tested" needs at least max(min_events, 2) of them that are not
  all equal; otherwise "insufficient_events" or "zero_variance", with no t or p.
- mean_car is the RAW mean CAR (the market's move after the events), as R0 and the
  matcher read effect_car. direction only flips the position: position_mean_car =
  mean_car for long, -mean_car for short, and hit_rate is the share of events where the
  position made money. t_stat is the raw t (its p-value does not depend on direction).
- Every run_backtest call is one test in the correction family (agent.interim.harness),
  including untestable signals (at p = 1, as R0 counts them), except a duplicate: a
  signal with the same measurable events and holding period as one already tested
  (including a rerun) repeats that test, uses no budget and becomes its alias.
"""

import math
from datetime import date
from typing import Any, get_args

import pandas as pd
from scipy import stats

from agent.interim.data import breakdowns, prices_until
from agent.interim.spec import DIRECTIONS, OPS_BY_KIND, select
from agent.tools import ToolError
from evaluation.reference_car import event_cars

TESTED, INSUFFICIENT, ZERO_VARIANCE = "tested", "insufficient_events", "zero_variance"


def finite_cars(cars: pd.DataFrame) -> pd.DataFrame:
    """R0's filter: drop missing and infinite CARs."""
    car = cars["car"]
    return cars.loc[car.notna() & ~car.isin([float("inf"), float("-inf")])].copy()


def summarize_returns(cars: pd.DataFrame, min_events: int) -> dict[str, Any]:
    """R0System.summarize_returns, line for line."""
    returns = cars["car"].astype(float)
    valid = returns.notna() & returns.ne(float("inf")) & returns.ne(float("-inf"))
    returns = returns.loc[valid]
    result = {
        "n_events": len(returns),
        "mean_car": float(returns.mean()) if len(returns) else None,
        "t_stat": None,
        "p_value_raw": None,
        "status": INSUFFICIENT,
    }
    if len(returns) < max(min_events, 2):
        return result
    if returns.nunique() == 1:
        result["status"] = ZERO_VARIANCE
        return result
    test = stats.ttest_1samp(returns, popmean=0.0)
    result.update(t_stat=float(test.statistic), p_value_raw=float(test.pvalue), status=TESTED)
    return result


def group_stats(cars: pd.Series, sign: float) -> dict[str, Any]:
    """Stats of one breakdown group: no minimum, but a t-test needs 2 unequal CARs."""
    cars = cars.astype(float)
    n = len(cars)
    out = {"n_events": n, "mean_car": float(cars.mean()) if n else None, "t_stat": None,
           "p_value_raw": None, "hit_rate": float((sign * cars > 0).mean()) if n else None}
    if n >= 2 and cars.nunique() > 1:
        test = stats.ttest_1samp(cars, popmean=0.0)
        out.update(t_stat=float(test.statistic), p_value_raw=float(test.pvalue))
    return out


def _round(value: Any, digits: int) -> Any:
    return round(value, digits) if isinstance(value, float) and math.isfinite(value) else value


def rounded(result: dict[str, Any]) -> dict[str, Any]:
    """Model-facing copy: means to 5 places, t to 3, p to 5, hit rate to 3."""
    places = {"mean_car": 5, "position_mean_car": 5, "t_stat": 3, "p_value_raw": 5,
              "hit_rate": 3}
    return {k: _round(v, places[k]) if k in places else v for k, v in result.items()}


class InterimBacktester:
    def __init__(
        self,
        events: pd.DataFrame,
        prices: pd.DataFrame,
        benchmarks: pd.DataFrame,
        budget: int,
        holding_days: int,
        min_events: int,
        signal_spec: type,
        research_end: date | None = None,
    ):
        """events: usable rows in the research period (agent.interim.data.research_events)."""
        self.events = events
        self.prices = prices_until(prices, research_end)
        self.benchmarks = prices_until(benchmarks, research_end)
        self.budget = budget
        self.holding_days = holding_days
        self.min_events = min_events
        self.signal_spec = signal_spec
        self.tests_used = 0
        self.specs: dict[str, Any] = {}
        self.results: dict[str, pd.DataFrame] = {}  # signal_id -> its latest run's CARs
        self.attempts: list[dict[str, Any]] = []  # one per counted test, in order
        self.aliases: dict[str, str] = {}  # duplicate signal_id -> the signal it repeats
        self.duplicates = 0  # run_backtest calls answered as duplicates (not counted)
        self._tested: dict[tuple, str] = {}  # (measurable event_ids, holding_days) -> signal
        # (signal, breakdown, group) -> raw p (None without a t-test): every subgroup the
        # agent has seen, for the strict correction (measured, not enforced)
        self.viewed_groups: dict[tuple[str, str, str], float | None] = {}
        self._breakdowns = breakdowns(events)

    def define(self, spec: dict[str, Any]) -> str:
        signal_id = f"sig-{len(self.specs) + 1}"
        self.specs[signal_id] = self.signal_spec.model_validate(spec)
        return signal_id

    def _spec(self, signal_id: str):
        if signal_id not in self.specs:
            known = ", ".join(self.specs) or "none yet"
            raise ToolError(f"unknown signal_id {signal_id!r}. Defined signals: {known}")
        return self.specs[signal_id]

    def canonical(self, signal_id: str) -> str:
        """The signal whose test a duplicate stands for (itself if not a duplicate)."""
        return self.aliases.get(signal_id, signal_id)

    def run(self, signal_id: str) -> dict[str, Any]:
        """Run the event study. A signal whose measurable events and holding period equal an
        already-tested signal's (direction aside: the test is two-sided) is a duplicate: it
        gets the same statistics, uses no budget, is not added to the correction family and
        becomes an alias of that signal.

        TODO(member-4): apply this duplicate rule to every system once the official
        harness exists (R0 never repeats an event set, so it is unaffected today).
        """
        spec = self._spec(signal_id)
        selected = select(self.events, spec)
        cars = finite_cars(event_cars(self.prices, self.benchmarks, selected, spec.holding_days))
        key = (frozenset(cars["event_id"]), spec.holding_days)
        original = self._tested.get(key)
        if original is None and self.tests_used >= self.budget:
            raise ToolError(f"testing budget exhausted ({self.budget} backtests used)")
        if original is None:
            self.tests_used += 1
            self._tested[key] = signal_id
        else:
            self.aliases[signal_id] = original
            self.duplicates += 1
        self.results[signal_id] = cars
        summary = summarize_returns(cars, self.min_events)
        sign = 1.0 if spec.direction == "long" else -1.0
        mean = summary["mean_car"]
        car = cars["car"].astype(float)
        result = {
            "signal_id": signal_id,
            "status": summary["status"],
            "n_events": summary["n_events"],  # measurable events
            "n_selected": len(selected),
            "n_unmeasurable": len(selected) - summary["n_events"],
            "mean_car": mean,
            "position_mean_car": None if mean is None else sign * mean,
            "t_stat": summary["t_stat"],
            "p_value_raw": summary["p_value_raw"],
            "hit_rate": float((sign * car > 0).mean()) if len(car) else None,
            "event_ids": list(cars["event_id"]),  # all of them, measurable only (as R0)
            "tests_used": self.tests_used,
            "budget_remaining": self.budget - self.tests_used,
            "duplicate_of": original,
        }
        if original is None:
            self.attempts.append({
                "test_index": self.tests_used, "signal_id": signal_id,
                "spec": spec.model_dump(mode="json"),
                **{k: result[k] for k in ("status", "n_events", "mean_car", "t_stat",
                                          "p_value_raw", "event_ids")},
            })
        return result

    def breakdown(self, signal_id: str, by: str) -> dict[str, Any]:
        spec = self._spec(signal_id)
        if signal_id not in self.results:
            raise ToolError(f"run_backtest {signal_id!r} before inspecting it")
        if by not in self._breakdowns:
            raise ToolError(f"by must be one of {list(self._breakdowns)}")
        cars = self.results[signal_id]
        if by == "year":
            keys = pd.to_datetime(cars["first_tradeable_date"]).dt.year
        else:
            keys = cars["event_id"].map(self.events.set_index("event_id")[by])
        sign = 1.0 if spec.direction == "long" else -1.0
        stats_by_group = {str(k): group_stats(g["car"], sign) for k, g in cars.groupby(keys)}
        for group, s in stats_by_group.items():  # for the strict correction (harness)
            if s["n_events"] and s["mean_car"] is not None:
                self.viewed_groups[(self.canonical(signal_id), by, group)] = s["p_value_raw"]
        groups = {k: rounded(s) for k, s in stats_by_group.items()}
        return {"signal_id": signal_id, "by": by, "groups": groups}

    def signal_options(self) -> dict[str, Any]:
        event_types = get_args(self.signal_spec.model_fields["event_type"].annotation)
        return {
            "event_types": list(event_types),
            "operators": {kind: list(ops) for kind, ops in OPS_BY_KIND.items()},
            "holding_days": [self.holding_days],
            "directions": list(DIRECTIONS),
            "breakdowns": list(self._breakdowns),
        }

    def budget_status(self) -> dict[str, int]:
        return {"budget": self.budget, "tests_used": self.tests_used}
