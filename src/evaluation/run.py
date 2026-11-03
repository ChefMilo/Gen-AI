"""Experiment runner: plan, build universes, run every system on every universe, score.

    python -m evaluation.run --config configs/experiment.yaml --systems null,random
    python -m evaluation.run --config configs/experiment.yaml --dry-run  # plan + build only

Systems come from the config's systems: block (or --systems). Its mapping form can list
modules to import first, each registering its own systems (evaluation.systems):

    systems:
      names: [null, r0, r2]
      modules: [agent.systems]

Output under results/<experiment_id>/ (or --out-dir), which must be git-ignored:
- plan.json (evaluation.plan) and universes/<universe_id>/ (evaluation.universe_store);
- runs/<system>/<universe_id>/: findings.jsonl (one Finding per line) and
  metadata.json, written last, which marks the pair finished. A failed pair writes
  error.json (with the traceback) instead and is retried on the next run;
- summary/: per_universe, recall_breakdowns, truths (evaluation.metrics),
  system_summary, detection_curve, comparisons (evaluation.stats) and pairs (status
  and wall time of every system x universe pair), as CSV.

With resume, a pair is skipped when its metadata.json is there and was written for the
same universe cache key, so runs made on other data or another plan entry are redone.
One pair failing is logged and does not stop the others.

Each pair gets its own seed, from base_seed, the system name and the universe id; it
does not reveal the universe type (unlike the placebo seeds).
"""

import argparse
import hashlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from agent.finding import Finding
from evaluation.metrics import MetricsTables, UniverseResult, evaluate, metrics_table
from evaluation.plan import (
    EXPERIMENT_CONFIG_PATH,
    REPO_ROOT,
    ExperimentConfig,
    ExperimentPlan,
    UniverseEntry,
    load_experiment_config,
    plan_experiment,
    save_plan,
)
from evaluation.stats import compare_systems, detection_curve_data, system_summary
from evaluation.systems import (
    ResearchSystem,
    SystemConfigError,
    SystemRun,
    make_systems,
    make_view,
    resolve_system_names,
)
from evaluation.universe_record import UniverseRecord
from evaluation.universe_store import (
    FORMAT_VERSION,
    RECORD,
    BaseData,
    BuildStatus,
    build_universes,
    ensure_git_ignored,
    load_universe,
    universe_dir,
    write_json,
)

log = logging.getLogger(__name__)

RUNS_DIR, SUMMARY_DIR, PLAN_FILE = "runs", "summary", "plan.json"
FINDINGS, RUN_META, RUN_ERROR = "findings.jsonl", "metadata.json", "error.json"


def default_out_dir(cfg: ExperimentConfig) -> Path:
    return REPO_ROOT / "results" / cfg.experiment_id


def restrict_to_period(data: BaseData, start, end) -> BaseData:
    """Events with first_tradeable_date in [start, end]; prices and benchmarks up to end.

    Prices before start stay, as look-back history. None leaves that side open.
    """
    events, prices, benchmarks = data.events, data.prices, data.benchmarks
    day = pd.to_datetime(events["first_tradeable_date"])
    keep = pd.Series(True, index=events.index)
    if start is not None:
        keep &= day >= pd.Timestamp(start)
    if end is not None:
        keep &= day <= pd.Timestamp(end)
        prices = prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(end)]
        benchmarks = benchmarks[pd.to_datetime(benchmarks["date"]) <= pd.Timestamp(end)]
    return BaseData(prices.reset_index(drop=True), benchmarks.reset_index(drop=True),
                    events[keep].reset_index(drop=True))


def load_base_data(cfg: ExperimentConfig) -> BaseData:
    """Base prices, benchmarks and events per cfg.data, cut to the research period."""
    from universes.fake_events import make_fake_events

    cfg = cfg.resolved()
    source = cfg.data
    if source.source == "synthetic":
        from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

        tickers = [f"T{i:03d}" for i in range(source.n_tickers)]
        prices = make_synthetic_prices(tickers, source.start, source.end, seed=source.seed)
        benchmarks = make_synthetic_benchmarks(source.start, source.end, seed=source.seed)
    else:
        from store.prices import load_benchmarks, load_prices

        log.warning("real prices with FAKE events: src/store has no event loader yet")
        prices, benchmarks = load_prices(), load_benchmarks()
    events = make_fake_events(prices, seed=source.events_seed)
    return restrict_to_period(BaseData(prices, benchmarks, events), cfg.research_start,
                              cfg.research_end)


def pair_seed(base_seed: int, system: str, universe_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{system}:{universe_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def run_dir(out_dir: Path, system: str, universe_id: str) -> Path:
    return Path(out_dir) / RUNS_DIR / system / universe_id


def is_finished(path: Path, universe_key: str) -> bool:
    meta = path / RUN_META
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return data.get("status") == "ok" and data.get("universe_key") == universe_key


@dataclass(frozen=True)
class PairStatus:
    system: str
    universe_id: str
    universe_type: str
    status: Literal["ran", "skipped", "failed"]
    seconds: float
    n_findings: int | None = None
    error: str | None = None


@dataclass
class ExperimentOutcome:
    plan: ExperimentPlan
    builds: list[BuildStatus]
    pairs: list[PairStatus] = field(default_factory=list)
    tables: MetricsTables | None = None
    summary_files: dict[str, Path] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def _run_system(system: ResearchSystem, view, budget: int, seed: int,
                record: UniverseRecord) -> SystemRun:
    if getattr(system, "needs_truths", False):
        return system.run_with_truths(view, budget, seed, list(record.truths))
    return system.run(view, budget, seed)


def run_pair(
    system: ResearchSystem, universe, cfg: ExperimentConfig, out_dir: Path, universe_key: str,
) -> PairStatus:
    """Run one system on one universe and write its outputs; never raises."""
    entry = universe.entry
    path = run_dir(out_dir, system.name, entry.universe_id)
    path.mkdir(parents=True, exist_ok=True)
    for name in (RUN_META, FINDINGS, RUN_ERROR):
        (path / name).unlink(missing_ok=True)
    seed = pair_seed(cfg.base_seed, system.name, entry.universe_id)
    view = make_view(entry.universe_id, universe.prices, universe.benchmarks, universe.events,
                     cfg.research_start, cfg.research_end)
    start = time.perf_counter()
    try:
        result = _run_system(system, view, cfg.backtest_budget, seed, universe.record)
        seconds = time.perf_counter() - start
        metadata = {"wall_time_s": seconds, **result.metadata}
        lines = "".join(f.model_dump_json() + "\n" for f in result.findings)
        (path / FINDINGS).write_text(lines, encoding="utf-8")
        write_json({
            "status": "ok", "system": system.name, "universe_id": entry.universe_id,
            "session_id": result.session_id, "seed": seed, "budget": cfg.backtest_budget,
            "universe_key": universe_key, "n_findings": len(result.findings),
            "metadata": metadata, "finished_at": datetime.now(UTC).isoformat(),
        }, path / RUN_META)
        return PairStatus(system.name, entry.universe_id, entry.universe_type, "ran",
                          seconds, len(result.findings))
    except Exception as err:  # one pair failing must not stop the experiment
        seconds = time.perf_counter() - start
        log.exception("%s on %s failed", system.name, entry.universe_id)
        write_json({
            "status": "failed", "system": system.name, "universe_id": entry.universe_id,
            "seed": seed, "universe_key": universe_key, "error": repr(err),
            "traceback": traceback.format_exc(),
        }, path / RUN_ERROR)
        return PairStatus(system.name, entry.universe_id, entry.universe_type, "failed",
                          seconds, error=repr(err))


def run_systems(
    plan: ExperimentPlan, systems: list[ResearchSystem], data: BaseData, out_dir: Path,
    resume: bool = True,
) -> list[PairStatus]:
    """Every system on every universe, loading each universe once."""
    cfg = plan.config
    statuses = []
    for entry in plan.universes:
        key = entry.cache_key(plan.fingerprint, FORMAT_VERSION)
        todo = [s for s in systems
                if not (resume and is_finished(run_dir(out_dir, s.name, entry.universe_id), key))]
        for s in systems:
            if s not in todo:
                statuses.append(PairStatus(s.name, entry.universe_id, entry.universe_type,
                                           "skipped", 0.0))
        if not todo:
            continue
        universe = load_universe(out_dir, entry, plan.fingerprint, data)
        for system in todo:
            status = run_pair(system, universe, cfg, out_dir, key)
            log.info("%s %s on %s (%s): %.1fs", status.status, system.name, entry.universe_id,
                     entry.universe_type, status.seconds)
            statuses.append(status)
    return statuses


def read_findings(path: Path) -> list[Finding]:
    text = (path / FINDINGS).read_text(encoding="utf-8")
    return [Finding.model_validate_json(line) for line in text.splitlines() if line.strip()]


def read_record(out_dir: Path, entry: UniverseEntry) -> UniverseRecord:
    path = universe_dir(out_dir, entry.universe_id) / RECORD
    return UniverseRecord.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_runs(plan: ExperimentPlan, system_names: list[str],
                  out_dir: Path) -> list[UniverseResult]:
    """UniverseResults for every finished pair of these systems."""
    results = []
    for entry in plan.universes:
        key = entry.cache_key(plan.fingerprint, FORMAT_VERSION)
        record = None
        for name in system_names:
            path = run_dir(out_dir, name, entry.universe_id)
            if not is_finished(path, key):
                continue
            record = record or read_record(out_dir, entry)
            meta = json.loads((path / RUN_META).read_text(encoding="utf-8"))
            results.append(evaluate(name, record, read_findings(path), plan.config.jaccard_min,
                                    meta.get("metadata", {})))
    return results


def write_summary(results: list[UniverseResult], cfg: ExperimentConfig, pairs: list[PairStatus],
                  out_dir: Path) -> tuple[MetricsTables | None, dict[str, Path]]:
    summary = Path(out_dir) / SUMMARY_DIR
    summary.mkdir(parents=True, exist_ok=True)
    frames = {"pairs": pd.DataFrame([p.__dict__ for p in pairs],
                                    columns=list(PairStatus.__dataclass_fields__))}
    tables = None
    if results:
        tables = metrics_table(results)
        frames.update({
            "per_universe": tables.per_universe,
            "recall_breakdowns": tables.breakdowns,
            "truths": tables.truths,
            "system_summary": system_summary(tables, seed=cfg.base_seed, n_boot=cfg.n_boot),
            "detection_curve": detection_curve_data(tables, seed=cfg.base_seed,
                                                    n_boot=cfg.n_boot),
            "comparisons": compare_systems(tables, seed=cfg.base_seed, n_boot=cfg.n_boot),
        })
    else:
        log.warning("no finished pairs to evaluate")
    files = {}
    for name, df in frames.items():
        files[name] = summary / f"{name}.csv"
        df.to_csv(files[name], index=False)
    return tables, files


def run_experiment(
    cfg: ExperimentConfig,
    systems: list[ResearchSystem],
    out_dir: Path | None = None,
    resume: bool = True,
    data: BaseData | None = None,
    dry_run: bool = False,
) -> ExperimentOutcome:
    """Plan, build or reuse the universes, run the systems, and write the summary.

    data defaults to load_base_data(cfg). dry_run stops after the universes are built.
    """
    cfg = cfg.resolved()
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(cfg)
    ensure_git_ignored(out_dir)
    names = [s.name for s in systems]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate system names: {names}")
    seconds = {}

    start = time.perf_counter()
    data = data if data is not None else load_base_data(cfg)
    seconds["load_data"] = time.perf_counter() - start

    start = time.perf_counter()
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    save_plan(plan, out_dir / PLAN_FILE)
    seconds["plan"] = time.perf_counter() - start

    start = time.perf_counter()
    builds = build_universes(out_dir, plan.universes, plan.fingerprint, data)
    seconds["build"] = time.perf_counter() - start
    outcome = ExperimentOutcome(plan=plan, builds=builds, seconds=seconds)
    if dry_run:
        return outcome

    start = time.perf_counter()
    outcome.pairs = run_systems(plan, systems, data, out_dir, resume)
    seconds["run"] = time.perf_counter() - start

    start = time.perf_counter()
    results = evaluate_runs(plan, names, out_dir)
    outcome.tables, outcome.summary_files = write_summary(results, cfg, outcome.pairs, out_dir)
    seconds["evaluate"] = time.perf_counter() - start
    return outcome


def _report(outcome: ExperimentOutcome, out_dir: Path) -> str:
    built = [b for b in outcome.builds if b.status == "built"]
    lines = [
        f"plan: {len(outcome.plan.universes)} universes "
        f"({len(outcome.plan.by_type('planted'))} planted, "
        f"{len(outcome.plan.by_type('placebo'))} placebo, "
        f"{len(outcome.plan.by_type('real'))} real), "
        f"{len(outcome.plan.kept_specs)} kept specs -> {out_dir}",
        f"universes: {len(built)} built, {len(outcome.builds) - len(built)} reused, "
        f"{sum(b.bytes for b in outcome.builds) / 1e6:.1f} MB on disk",
        "times (s): " + ", ".join(f"{k} {v:.1f}" for k, v in outcome.seconds.items()),
    ]
    if outcome.pairs:
        counts = pd.Series([p.status for p in outcome.pairs]).value_counts().to_dict()
        lines.append("pairs: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if outcome.tables is not None:
        summary = system_summary(outcome.tables, n_boot=1)
        cols = ["system", "recall", "precision", "mean_false_discoveries"]
        lines.append(summary[cols].to_string(index=False))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--systems", help="comma-separated system names; default: config")
    parser.add_argument("--out-dir", type=Path, help="default: results/<experiment_id>")
    parser.add_argument("--dry-run", action="store_true", help="plan and build universes only")
    parser.add_argument("--no-resume", action="store_true", help="rerun finished pairs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_experiment_config(args.config).resolved()
    names = args.systems.split(",") if args.systems else cfg.system_names
    names = [n.strip() for n in names if n.strip()]
    try:
        resolve_system_names(names, cfg.system_modules)
    except SystemConfigError as err:
        parser.error(str(err))
    systems = make_systems(names, cfg.holding_days)
    out_dir = args.out_dir or default_out_dir(cfg)
    outcome = run_experiment(cfg, systems, out_dir, resume=not args.no_resume,
                             dry_run=args.dry_run)
    print(_report(outcome, out_dir))
    return 1 if any(p.status == "failed" for p in outcome.pairs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
