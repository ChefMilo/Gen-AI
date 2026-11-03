"""Offline oracle check of a final plan (evaluation.oracle). No LLM, no API calls.

    python scripts/oracle_check.py                      # configs/final.yaml
    python scripts/oracle_check.py --config configs/final.yaml --out-dir results/final-v1

Builds (or reuses) the plan and universes the way `python -m evaluation.run --dry-run`
does, writes <out-dir>/oracle.csv and prints the findable rate by strength x
complexity x feature type.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.oracle import (  # noqa: E402
    findable_rates,
    judging_rules,
    monotone_violations,
    oracle_table,
)
from evaluation.plan import REPO_ROOT, load_experiment_config  # noqa: E402
from evaluation.run import default_out_dir, load_base_data, run_experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "final.yaml")
    parser.add_argument("--out-dir", type=Path, help="default: results/<experiment_id>")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = load_experiment_config(args.config).resolved()
    out_dir = args.out_dir or default_out_dir(cfg)
    data = load_base_data(cfg)
    outcome = run_experiment(cfg, [], out_dir, data=data, dry_run=True)
    alpha, min_events = judging_rules()
    table = oracle_table(outcome.plan, data, out_dir, alpha, min_events)
    path = out_dir / "oracle.csv"
    table.to_csv(path, index=False)
    print(f"{len(table)} planted (spec, universe) pairs -> {path}")
    print(f"family: Holm over backtest_budget={cfg.backtest_budget}; alpha={alpha}; "
          f"min_events={min_events}")
    print(f"findable overall: {table['findable'].mean():.1%}")
    print(findable_rates(table).to_string(index=False, float_format="{:.2f}".format))
    bad = monotone_violations(table)
    if bad.empty:
        print("findability is monotone in strength in every cell")
    else:
        print("NOT MONOTONE in strength (findable rate below a weaker strength's):")
        print(bad.to_string(index=False, float_format="{:.2f}".format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
