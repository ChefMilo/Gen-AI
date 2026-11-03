"""Oracle check: can each planted cell be found at all, under the systems' own rules?

For every planted spec in every planted universe of a plan built from a grid
(evaluation.final_grid), run the exact SignalSpec that selects its targets through the
same event study the harness uses (backtest.engine.run_event_study, on the view a
system sees), and correct its p-value the way the systems are judged:

- Holm (harness.corrections.holm_adjust) over a family of backtest_budget tests, with
  the oracle's test the only real one (the rest count as p = 1). That is
  min(1, budget * p_raw): the best a guarded session that spent its budget can get,
  never more lenient than the systems;
- findable: p_corrected < alpha (alpha and min_events from configs/default.yaml).

A cell that is not findable for the oracle cannot count against a system's recall in
any fair reading; the findable rate per strength x complexity x feature type shows
whether the grid's weak cells are detectable at all.
"""

from pathlib import Path

import pandas as pd
import yaml

from backtest.engine import run_event_study
from evaluation.final_grid import build_grid
from evaluation.plan import DEFAULT_CONFIG_PATH, ExperimentPlan
from evaluation.systems import make_view
from evaluation.universe_store import BaseData, load_universe
from harness.corrections import holm_adjust

COLUMNS = ["spec_id", "universe_id", "strength", "complexity", "feature_type", "n_events",
           "t", "p_raw", "p_corrected", "findable"]
GROUPS = ["strength", "complexity", "feature_type"]


def judging_rules(path: Path = DEFAULT_CONFIG_PATH) -> tuple[float, int]:
    """(alpha, min_events) the systems are judged with, from configs/default.yaml."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return float(cfg["alpha"]), int(cfg["min_events"])


def oracle_table(plan: ExperimentPlan, data: BaseData, out_dir: Path,
                 alpha: float | None = None, min_events: int | None = None) -> pd.DataFrame:
    """One row per (planted universe, spec). The universes must be built (dry run)."""
    cfg = plan.config
    if cfg.grid is None:
        raise ValueError("the oracle check needs a plan built from a grid: block")
    default_alpha, default_min = judging_rules()
    alpha = default_alpha if alpha is None else alpha
    min_events = default_min if min_events is None else min_events
    budget = cfg.backtest_budget
    signals = {c.plant.spec_id: c.signal for c in build_grid(cfg.grid, cfg.holding_days)}
    rows = []
    for entry in plan.by_type("planted"):
        universe = load_universe(out_dir, entry, plan.fingerprint, data)
        view = make_view(entry.universe_id, universe.prices, universe.benchmarks,
                         universe.events, cfg.research_start, cfg.research_end)
        for spec in entry.specs:
            result = run_event_study(view.prices, view.benchmarks, view.events,
                                     signals[spec.spec_id], f"oracle-{spec.spec_id}",
                                     min_events=min_events)
            tested = result.status == "tested"
            p_raw = result.p_value_raw if tested else 1.0
            p_corrected = holm_adjust([p_raw], family_size=max(budget, 1))[0]
            rows.append({
                "spec_id": spec.spec_id, "universe_id": entry.universe_id,
                "strength": spec.strength, "complexity": spec.complexity,
                "feature_type": spec.feature_type, "n_events": result.n_events,
                "t": result.t_stat if tested else float("nan"), "p_raw": p_raw,
                "p_corrected": p_corrected, "findable": bool(p_corrected < alpha),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def findable_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Findable share and count per strength x complexity x feature type."""
    out = (table.groupby(GROUPS, sort=True)["findable"]
           .agg(findable_rate="mean", findable="sum", n="count").reset_index())
    out["findable"] = out["findable"].astype(int)
    return out
