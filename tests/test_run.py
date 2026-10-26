"""run_experiment: resume, per-pair failures, end-to-end scores and the CLI."""

import json

import pandas as pd
import pytest

from evaluation.run import FINDINGS, RUN_ERROR, RUN_META, SUMMARY_DIR, main, run_dir, run_experiment
from evaluation.systems import NullSystem, OracleSystem, RandomSystem, SystemRun
from evaluation.universe_store import BaseData
from experiment_helpers import config, small_data

SUMMARY_FILES = {"pairs", "per_universe", "recall_breakdowns", "truths", "system_summary",
                 "detection_curve", "comparisons"}


class FailingSystem:
    """Raises on one universe, finds nothing elsewhere."""

    def __init__(self, fail_on: str | None, name: str = "failing"):
        self.fail_on, self.name = fail_on, name

    def run(self, view, budget, seed):
        if view.universe_id == self.fail_on:
            raise RuntimeError("boom")
        return SystemRun(findings=[], session_id=f"{self.name}-{view.universe_id}")


@pytest.fixture(scope="module")
def end_to_end(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    systems = [OracleSystem(), NullSystem(), RandomSystem(min_events=20)]
    outcome = run_experiment(config(), systems, out, data=small_data(noise=False))
    return out, outcome


@pytest.mark.slow
def test_oracle_recovers_every_truth_and_null_none(end_to_end):
    _, outcome = end_to_end
    assert {p.status for p in outcome.pairs} == {"ran"}
    df = outcome.tables.per_universe
    planted = df[df["universe_type"] == "planted"]
    placebo = df[df["universe_type"] == "placebo"]
    oracle = planted[planted["system"] == "oracle"]
    assert oracle["n_matched"].sum() == oracle["n_truths"].sum() > 0  # recall 1 without noise
    assert (oracle["effect_error_mean_abs"] < 1e-9).all()
    assert (placebo.loc[placebo["system"] == "oracle", "false_discoveries"] == 0).all()
    null = df[df["system"] == "null"]
    assert (null.loc[null["universe_type"] == "planted", "recall"] == 0).all()
    assert (null.loc[null["universe_type"] == "placebo", "false_discoveries"] == 0).all()
    assert len(df) == 3 * len(outcome.plan.universes)


@pytest.mark.slow
def test_summary_csvs_are_written(end_to_end):
    out, outcome = end_to_end
    assert set(outcome.summary_files) == SUMMARY_FILES
    for name in SUMMARY_FILES:
        assert (out / SUMMARY_DIR / f"{name}.csv").exists()
    # By default read_csv takes the system name "null" for a missing value.
    summary = pd.read_csv(out / SUMMARY_DIR / "system_summary.csv",
                          keep_default_na=False, na_values=[""]).set_index("system")
    assert summary.loc["oracle", "recall"] == 1.0
    assert summary.loc["null", "recall"] == 0.0
    assert summary.loc["oracle", "mean_false_discoveries"] == 0.0
    comparisons = pd.read_csv(out / SUMMARY_DIR / "comparisons.csv", keep_default_na=False,
                              na_values=[""])
    recall = comparisons.query("system_a == 'oracle' and system_b == 'null' and "
                               "metric == 'recall'").iloc[0]
    assert recall["estimate"] == 1.0 and recall["lo"] == 1.0
    curve = pd.read_csv(out / SUMMARY_DIR / "detection_curve.csv")
    # Zero-noise prices: baseline std 0, so no snr and no snr_bin rows.
    assert set(curve["group"]) == {"strength", "feature_type", "complexity"}


@pytest.mark.slow
def test_findings_and_metadata_files(end_to_end):
    out, outcome = end_to_end
    entry = outcome.plan.by_type("planted")[0]
    path = run_dir(out, "oracle", entry.universe_id)
    meta = json.loads((path / RUN_META).read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["n_findings"] == len(entry.specs)
    assert meta["metadata"]["wall_time_s"] >= 0
    lines = (path / FINDINGS).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(entry.specs)


def small_config(**overrides):
    return config(n_planted=0, n_placebo=2, **overrides)


def test_resume_skips_finished_pairs(tmp_path):
    data = small_data()
    systems = [NullSystem(), RandomSystem(min_events=20)]
    first = run_experiment(small_config(), systems, tmp_path, data=data)
    assert {p.status for p in first.pairs} == {"ran"}
    findings = run_dir(tmp_path, "random", first.plan.universes[0].universe_id) / FINDINGS
    written = findings.stat().st_mtime_ns

    second = run_experiment(small_config(), systems, tmp_path, data=data)
    assert {p.status for p in second.pairs} == {"skipped"}
    assert {b.status for b in second.builds} == {"reused"}
    assert findings.stat().st_mtime_ns == written
    pd.testing.assert_frame_equal(second.tables.per_universe, first.tables.per_universe)

    third = run_experiment(small_config(), systems, tmp_path, resume=False, data=data)
    assert {p.status for p in third.pairs} == {"ran"}

    events = data.events.iloc[1:].reset_index(drop=True)  # other data: universes and runs redo
    fourth = run_experiment(small_config(), systems, tmp_path,
                            data=BaseData(data.prices, data.benchmarks, events))
    assert {b.status for b in fourth.builds} == {"built"}
    assert {p.status for p in fourth.pairs} == {"ran"}


def test_one_failing_pair_does_not_stop_the_rest(tmp_path):
    data = small_data()
    plan_ids = [u.universe_id for u in
                run_experiment(small_config(), [], tmp_path, data=data, dry_run=True)
                .plan.universes]
    bad = plan_ids[1]
    systems = [FailingSystem(bad), NullSystem()]
    outcome = run_experiment(small_config(), systems, tmp_path, data=data)
    status = {(p.system, p.universe_id): p.status for p in outcome.pairs}
    assert status.pop(("failing", bad)) == "failed"
    assert set(status.values()) == {"ran"}
    error = json.loads((run_dir(tmp_path, "failing", bad) / RUN_ERROR).read_text("utf-8"))
    assert "boom" in error["error"] and "RuntimeError" in error["traceback"]
    assert not (run_dir(tmp_path, "failing", bad) / RUN_META).exists()
    scored = outcome.tables.per_universe
    assert len(scored) == 2 * len(plan_ids) - 1
    assert ((scored["system"] == "failing") & (scored["universe_id"] == bad)).sum() == 0
    pairs = pd.read_csv(tmp_path / SUMMARY_DIR / "pairs.csv")
    assert (pairs["status"] == "failed").sum() == 1

    fixed = run_experiment(small_config(), [FailingSystem(None), NullSystem()], tmp_path,
                           data=data)
    status = {(p.system, p.universe_id): p.status for p in fixed.pairs}
    assert status.pop(("failing", bad)) == "ran"  # retried
    assert set(status.values()) == {"skipped"}
    assert not (run_dir(tmp_path, "failing", bad) / RUN_ERROR).exists()


def test_cli_dry_run_plans_and_builds_only(tmp_path, capsys):
    cfg = tmp_path / "experiment.yaml"
    cfg.write_text(
        "experiment_id: cli\nn_planted: 0\nn_placebo: 1\nsystems: ['null']\nn_boot: 10\n"
        "data:\n  n_tickers: 10\n  start: 2015-01-01\n  end: 2018-12-31\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert main(["--config", str(cfg), "--out-dir", str(out), "--dry-run"]) == 0
    assert (out / "plan.json").exists()
    assert len(list((out / "universes").iterdir())) == 2  # placebo + real
    assert not (out / "runs").exists()
    assert "2 built" in capsys.readouterr().out

    assert main(["--config", str(cfg), "--out-dir", str(out), "--systems", "null,random"]) == 0
    assert (out / SUMMARY_DIR / "system_summary.csv").exists()
    assert "2 reused" in capsys.readouterr().out
