"""Run the agent systems (and any registered ones) through the team's experiment runner.

    python -m agent.experiment --config configs/experiment_agent_dev.yaml --systems null,random
    python -m agent.experiment --config configs/experiment_agent_dev.yaml --systems r0,r2,r3
    python -m agent.experiment --config ... --systems r2,r3 --out results/dev_agent_v2

The runner's own CLI (python -m evaluation.run) does not import agent code, so it cannot
see the agent systems; this entry point registers them and then drives the runner's
public functions (load_base_data, build_universes, run_systems, evaluate_runs,
write_summary), editing nothing in src/evaluation. Once the runner can import system
modules from its config, this file can go away.

The config file has two parts:
- experiment: an evaluation.plan.ExperimentConfig (data, budget, holding days, systems);
- dev: the universes, built from the DEV plant specs and seeds listed there instead of
  default_spec_grid. Dev specs that are in the final grid, or seeds the final plan in
  configs/experiment.yaml uses, are refused.

Agent traces go to <out_dir>/traces. Agent systems get the plan's universe_id ->
universe_type map privately, only to refuse real universes without research_end.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent.config import load_config
from agent.system import AGENT_SYSTEMS
from evaluation import systems as runner_systems
from evaluation.plan import (
    EXPERIMENT_CONFIG_PATH,
    PLACEBO_SEED_OFFSET,
    ExperimentConfig,
    ExperimentPlan,
    UniverseEntry,
    data_fingerprint,
    load_experiment_config,
    save_plan,
)
from evaluation.run import (
    PLAN_FILE,
    ExperimentOutcome,
    default_out_dir,
    evaluate_runs,
    load_base_data,
    run_systems,
    write_summary,
)
from evaluation.stats import system_summary
from evaluation.universe_store import BaseData, build_universes, ensure_git_ignored
from universes.placebo import DEFAULT_MIN_GAP_DAYS
from universes.planted import PlantSpec, candidate_specs

log = logging.getLogger(__name__)
TRACES_DIR = "traces"


class DevPlanted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    specs: list[PlantSpec] = Field(min_length=1)


class DevPlacebo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int


class DevUniverses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    universe_id_prefix: str = Field("dev_", pattern=r"^dev_")
    planted: list[DevPlanted] = []
    placebo: list[DevPlacebo] = []


class DevExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    dev: DevUniverses


def load_dev_config(path: Path) -> DevExperimentConfig:
    with open(path, encoding="utf-8") as f:
        return DevExperimentConfig.model_validate(yaml.safe_load(f))


def register_agent_systems() -> None:
    """Make r2, r3 and r3-u known to evaluation.systems.make_systems. Built that way they
    have no universe-type map, so they refuse to run without research_end."""
    runner_systems.SYSTEMS.update(AGENT_SYSTEMS)


def _spec_key(spec: PlantSpec) -> tuple:
    text = spec.text_feature.model_dump_json() if spec.text_feature else None
    return (spec.event_type, spec.condition, text, spec.effect, spec.direction)


def check_dev_only(cfg: DevExperimentConfig, events: pd.DataFrame,
                   final_path: Path = EXPERIMENT_CONFIG_PATH) -> None:
    """Raise if a dev spec is in the final spec grid or a dev seed is a final-plan seed."""
    final = load_experiment_config(final_path).resolved()
    grid = {_spec_key(s) for s in candidate_specs(events, final.holding_days)}
    for planted in cfg.dev.planted:
        for spec in planted.specs:
            if _spec_key(spec) in grid:
                raise ValueError(f"dev spec {spec.spec_id} is in the final spec grid")
    final_seeds = set(range(final.base_seed, final.base_seed + final.n_planted))
    offset = final.base_seed + PLACEBO_SEED_OFFSET
    final_seeds |= set(range(offset, offset + final.n_placebo))
    used = [u.seed for u in [*cfg.dev.planted, *cfg.dev.placebo]] + [cfg.experiment.base_seed]
    clash = sorted(set(used) & final_seeds)
    if clash:
        raise ValueError(f"dev seeds {clash} are used by the final plan ({final_path.name})")


def build_dev_plan(cfg: DevExperimentConfig, data: BaseData) -> ExperimentPlan:
    """An ExperimentPlan over the dev universes, in a seeded order of opaque ids."""
    exp = cfg.experiment.resolved()
    kinds = [("planted", p) for p in cfg.dev.planted] + [("placebo", p) for p in cfg.dev.placebo]
    order = np.random.default_rng([exp.base_seed, 11]).permutation(len(kinds))
    entries = []
    for k, i in enumerate(order, start=1):
        kind, dev = kinds[i]
        universe_id = f"{cfg.dev.universe_id_prefix}u{k:02d}"
        if kind == "planted":
            entries.append(UniverseEntry(
                universe_id=universe_id, universe_type="planted", seed=dev.seed,
                holding_days=exp.holding_days, min_events=exp.plant_min_events,
                specs=[s.model_copy(update={"seed": dev.seed}) for s in dev.specs]))
        else:
            entries.append(UniverseEntry(
                universe_id=universe_id, universe_type="placebo", seed=dev.seed,
                holding_days=exp.holding_days,
                min_gap_days=exp.min_gap_days or DEFAULT_MIN_GAP_DAYS))
    specs = [s for e in entries for s in e.specs]
    return ExperimentPlan(
        experiment_id=exp.experiment_id, base_seed=exp.base_seed, config=exp,
        fingerprint=data_fingerprint(data.prices, data.benchmarks, data.events),
        kept_specs=specs, dropped_specs=[], appearances={s.spec_id: 1 for s in specs},
        planned_clashes={}, universes=entries,
    )


def make_systems(names: list[str], plan: ExperimentPlan, config: dict, **agent_kwargs):
    """Agent systems with the plan's private universe-type map; the rest via the runner."""
    types = {e.universe_id: e.universe_type for e in plan.universes}
    holding_days = plan.config.holding_days
    out = []
    for name in names:
        if name in AGENT_SYSTEMS:
            out.append(AGENT_SYSTEMS[name](config=config, holding_days=holding_days,
                                           universe_types=types, **agent_kwargs))
        else:
            out.extend(runner_systems.make_systems([name], holding_days))
    return out


def run_dev_experiment(
    cfg: DevExperimentConfig,
    names: list[str],
    out_dir: Path | None = None,
    resume: bool = True,
    data: BaseData | None = None,
    dry_run: bool = False,
    agent_config: dict | None = None,
    **agent_kwargs,
) -> ExperimentOutcome:
    exp = cfg.experiment.resolved()
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(exp)
    ensure_git_ignored(out_dir)
    data = data if data is not None else load_base_data(exp)
    check_dev_only(cfg, data.events)
    plan = build_dev_plan(cfg, data)
    save_plan(plan, out_dir / PLAN_FILE)
    builds = build_universes(out_dir, plan.universes, plan.fingerprint, data)
    outcome = ExperimentOutcome(plan=plan, builds=builds)
    if dry_run:
        return outcome

    config = agent_config if agent_config is not None else load_config()
    config = {**config, "agent": {**config["agent"], "trace_dir": out_dir / TRACES_DIR}}
    systems = make_systems(names, plan, config, **agent_kwargs)
    outcome.pairs = run_systems(plan, systems, data, out_dir, resume)
    results = evaluate_runs(plan, [s.name for s in systems], out_dir)
    outcome.tables, outcome.summary_files = write_summary(results, exp, outcome.pairs, out_dir)
    return outcome


def report(outcome: ExperimentOutcome, out_dir: Path) -> str:
    plan = outcome.plan
    lines = [f"plan: {len(plan.universes)} dev universes "
             f"({len(plan.by_type('planted'))} planted, {len(plan.by_type('placebo'))} "
             f"placebo), specs {[s.spec_id for s in plan.kept_specs]} -> {out_dir}"]
    for p in outcome.pairs:
        extra = (f" error={p.error}" if p.error
                 else f" findings={p.n_findings}" if p.n_findings is not None else "")
        lines.append(f"  {p.system:7s} {p.universe_id} {p.status} {p.seconds:.1f}s{extra}")
    if outcome.tables is not None:
        cols = ["system", "recall", "precision", "mean_false_discoveries"]
        lines.append(system_summary(outcome.tables, n_boot=1)[cols].to_string(index=False))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.experiment", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="a dev experiment config")
    parser.add_argument("--systems", help="comma-separated system names; default: config")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, metavar="DIR",
                        help="output folder (default: results/<experiment_id>)")
    parser.add_argument("--dry-run", action="store_true", help="plan and build universes only")
    parser.add_argument("--no-resume", action="store_true", help="rerun finished pairs")
    parser.add_argument("--model", help="agent model for this run (overrides AGENT_MODEL)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_dev_config(args.config)
    exp = cfg.experiment.resolved()
    names = [n.strip() for n in (args.systems.split(",") if args.systems
                                 else exp.systems) if n.strip()]
    register_agent_systems()
    unknown = [n for n in names if n not in runner_systems.SYSTEMS]
    if unknown:
        parser.error(f"unknown systems {unknown}; known: {sorted(runner_systems.SYSTEMS)}"
                     " (r0 needs PR #17)")
    config = load_config()
    if any(n in AGENT_SYSTEMS for n in names):
        from agent.run_session import load_dotenv_if_available

        load_dotenv_if_available()
        config = load_config()  # $AGENT_MODEL may come from .env
        if args.model:
            config["agent"]["model"] = args.model
    out_dir = args.out_dir or default_out_dir(exp)
    outcome = run_dev_experiment(cfg, names, out_dir, resume=not args.no_resume,
                                 dry_run=args.dry_run, agent_config=config)
    print(report(outcome, out_dir))
    return 1 if any(p.status == "failed" for p in outcome.pairs) else 0


if __name__ == "__main__":
    sys.exit(main())
