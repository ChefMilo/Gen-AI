"""Prompt versions of the LLM systems on every universe of a plan (no LLM calls).

    python -m agent.freeze --config configs/final.yaml            # print versions
    python -m agent.freeze --config configs/final.yaml --check    # all frozen?

For each LLM system in the config (r1, r2, r3, r3-u), builds every universe of the plan
in memory, prepares the session exactly as a run would (system.prepare: ports, brief,
tool schemas) at the plan's budget, and computes its prompt_version. Without --check it
prints the YAML for agent.frozen_prompt_versions in configs/default.yaml; with --check
it exits non-zero unless every session would pass the freeze check.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from agent.config import load_config
from evaluation import systems as runner_systems
from evaluation.plan import load_experiment_config, plan_experiment
from evaluation.run import load_base_data
from evaluation.systems import make_view
from evaluation.universe_store import build_universe

LLM_SYSTEMS = ("r1", "r2", "r3", "r3-u")


def plan_prompt_versions(config_path: Path,
                         universe_ids: list[str] | None = None) -> dict[str, dict[str, str]]:
    """system -> universe_id -> prompt_version, for the config's LLM systems (all
    universes, or only universe_ids)."""
    import agent.systems  # noqa: F401  (registers the LLM systems)

    cfg = load_experiment_config(config_path).resolved()
    names = [n for n in cfg.system_names if n in LLM_SYSTEMS]
    data = load_base_data(cfg)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    systems = runner_systems.make_systems(names, cfg.holding_days)  # as evaluation.run does
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for entry in plan.universes:
        if universe_ids is not None and entry.universe_id not in universe_ids:
            continue
        universe = build_universe(entry, data)
        view = make_view(entry.universe_id, universe.prices, universe.benchmarks,
                         universe.events, cfg.research_start, cfg.research_end)
        for system in systems:
            prepared = system.prepare(view, cfg.backtest_budget)
            out[system.name][entry.universe_id] = prepared[-1]
    return dict(out)


def frozen_yaml(versions: dict[str, dict[str, str]]) -> str:
    distinct = {name: sorted(set(by_universe.values())) for name, by_universe in versions.items()}
    return yaml.safe_dump({"frozen_prompt_versions": distinct}, sort_keys=False)


def unfrozen(versions: dict[str, dict[str, str]], agent: dict) -> list[tuple[str, str, str]]:
    """(system, universe_id, version) for every session the freeze check would refuse."""
    frozen = agent.get("frozen_prompt_versions") or {}
    return [(name, uid, v) for name, by_universe in versions.items()
            for uid, v in by_universe.items() if v not in (frozen.get(name) or [])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.freeze", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="an experiment config")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 unless every session's prompt_version is frozen")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    versions = plan_prompt_versions(args.config)
    for name, by_universe in versions.items():
        print(f"{name}: {len(by_universe)} universes, versions {sorted(set(by_universe.values()))}")
    if not args.check:
        print(frozen_yaml(versions))
        return 0
    missing = unfrozen(versions, load_config()["agent"])
    for name, uid, version in missing:
        print(f"NOT FROZEN: {name} on {uid}: {version}")
    print("freeze check:", "FAILED" if missing else
          f"passed ({sum(len(v) for v in versions.values())} sessions)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
