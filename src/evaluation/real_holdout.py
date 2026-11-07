"""Holdout check for the exploratory real-data run (configs/real_explore.yaml).

Each system's significant findings on the real research-period universe are re-tested,
exactly as reported (signal spec fixed), on the holdout period: holdout_start in
configs/default.yaml to the last price date. harness.holdout.evaluate_holdout_once does
the test and keeps one ledger per system: it reserves the ledger before it touches the
holdout data and refuses a second run, so the holdout is looked at once.

    python -m evaluation.real_holdout --config configs/real_explore.yaml [--systems r0,r2]

Needs $GENAI_DATA_DIR. Writes <results>/holdout/<system>.json and prints a table.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from evaluation.plan import DEFAULT_CONFIG_PATH, load_experiment_config, load_plan
from evaluation.run import PLAN_FILE, default_out_dir, read_findings, run_dir

HOLDOUT_DIR = "holdout"


def holdout_start() -> date:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return pd.Timestamp(yaml.safe_load(f)["holdout_start"]).date()


def store_loader(start: date, end: date):
    """(prices, benchmarks, events) from the Data folder: usable events from start to end."""
    from store.events import load_events
    from store.prices import load_benchmarks, load_prices

    events = load_events(start, end)
    prices = load_prices(set(events["ticker"]), start, end)
    return prices, load_benchmarks(), events


def last_price_date() -> date:
    from store.prices import load_benchmarks

    return pd.Timestamp(load_benchmarks()["date"].max()).date()


def check_system(out_dir: Path, system: str, real_id: str, config, loader) -> dict:
    """The holdout report for one system (or why there is none)."""
    from harness.holdout import evaluate_holdout_once

    path = run_dir(out_dir, system, real_id)
    if not path.is_dir():
        return {"system": system, "status": "not run"}
    significant = [f for f in read_findings(path) if f.passed_correction]
    if not significant:
        return {"system": system, "status": "no significant findings to test"}
    ledger = out_dir / HOLDOUT_DIR / f"{system}.json"
    if ledger.exists():
        report = json.loads(ledger.read_text(encoding="utf-8"))
        return {**report, "system": system, "note": "already run; showing the ledger"}
    return {**evaluate_holdout_once(significant, config, ledger, loader), "system": system}


def table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        for row in report.get("results", []):
            rows.append({"system": report["system"], **row})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    from harness.holdout import HoldoutConfig

    parser = argparse.ArgumentParser(prog="python -m evaluation.real_holdout",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--systems", help="comma-separated; default: the config's systems")
    parser.add_argument("--out-dir", type=Path, help="default: results/<experiment_id>")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config).resolved()
    if not (cfg.include_real and cfg.data.real == "store" and cfg.data.real_period == "research"):
        parser.error("the config's real universe must be data.real: store, research period")
    out_dir = args.out_dir or default_out_dir(cfg)
    plan = load_plan(out_dir / PLAN_FILE)
    [real] = plan.by_type("real")
    names = args.systems.split(",") if args.systems else cfg.system_names
    config = HoldoutConfig(research_end=cfg.research_end, holdout_start=holdout_start(),
                           holdout_end=last_price_date())  # min_events 20, alpha 0.05
    reports = [check_system(out_dir, n.strip(), real.universe_id, config, store_loader)
               for n in names if n.strip()]
    for report in reports:
        if not report.get("results"):
            print(f"{report['system']}: {report.get('status')}")
    df = table(reports)
    if not df.empty:
        cols = ["system", "signal_id", "research_effect_car", "holdout_effect_car", "n_events",
                "p_value_corrected", "passed_correction", "same_sign", "retention_ratio"]
        print(df[cols].to_string(index=False))
        df.to_csv(out_dir / HOLDOUT_DIR / "holdout_results.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
