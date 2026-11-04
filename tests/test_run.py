"""run_experiment: resume, per-pair failures, end-to-end scores and the CLI."""

import json

import pandas as pd
import pytest

from evaluation import summarize
from evaluation.plan import load_plan, plan_sha256
from evaluation.run import (
    FINDINGS,
    PLAN_FILE,
    RUN_ERROR,
    RUN_META,
    SUMMARY_DIR,
    main,
    run_dir,
    run_experiment,
)
from evaluation.summary import INDEX_FILE, read_summary
from evaluation.systems import NullSystem, OracleSystem, RandomSystem, SystemRun
from evaluation.universe_store import BaseData
from experiment_helpers import config, small_data

SUMMARY_FILES = {"pairs", "per_universe", "recall_breakdowns", "truths", "system_summary",
                 "detection_curve", "detection_cells", "comparisons", "group_comparisons",
                 "truth_comparisons"}


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
    summary = read_summary(out / SUMMARY_DIR, "system_summary").set_index("system")
    assert summary.loc["oracle", "recall"] == 1.0
    assert summary.loc["null", "recall"] == 0.0
    assert summary.loc["oracle", "mean_false_discoveries"] == 0.0
    comparisons = read_summary(out / SUMMARY_DIR, "comparisons")
    # Pairs are in reporting order: stubs alphabetically.
    recall = comparisons.query("system_a == 'null' and system_b == 'oracle' and "
                               "metric == 'recall'").iloc[0]
    assert recall["estimate"] == -1.0 and recall["hi"] == -1.0
    curve = read_summary(out / SUMMARY_DIR, "detection_curve")
    # Zero-noise prices: baseline std 0, so no snr and no snr_bin rows.
    assert set(curve["group"]) == {"strength", "feature_type", "complexity"}
    group = read_summary(out / SUMMARY_DIR, "truth_comparisons")
    assert {"a_only", "b_only", "both", "neither", "p_value"} <= set(group.columns)
    assert read_summary(out / SUMMARY_DIR, "group_comparisons")["n_universes"].max() > 0


@pytest.mark.slow
def test_index_describes_the_summary(end_to_end):
    out, outcome = end_to_end
    index = json.loads((out / SUMMARY_DIR / INDEX_FILE).read_text(encoding="utf-8"))
    assert index["experiment_id"] == "test"
    assert index["plan_sha256"] == outcome.plan_sha256
    assert index["systems"] == ["null", "oracle", "random"]
    assert index["universes"] == {"planted": len(outcome.plan.by_type("planted")),
                                  "placebo": len(outcome.plan.by_type("placebo")),
                                  "real": len(outcome.plan.by_type("real"))}
    assert set(index["files"]) == {f"{name}.csv" for name in SUMMARY_FILES}
    env = index["environment"]
    assert {"python", "numpy", "pandas", "cpu_simd"} <= set(env)
    assert isinstance(env["cpu_simd"]["found"], list)
    assert index["created_at"].endswith("+00:00")


def test_run_with_one_system_keeps_the_others_in_the_summary(tmp_path):
    data = small_data()
    run_experiment(small_config(), [NullSystem(), RandomSystem(min_events=20)], tmp_path,
                   data=data)
    outcome = run_experiment(small_config(), [NullSystem()], tmp_path, data=data)
    assert {p.system for p in outcome.pairs} == {"null"}
    scored = read_summary(tmp_path / SUMMARY_DIR, "per_universe")
    assert set(scored["system"]) == {"null", "random"}
    pairs = read_summary(tmp_path / SUMMARY_DIR, "pairs")
    assert set(pairs["system"]) == {"null", "random"}
    assert set(pairs["status"]) == {"ran"}
    assert (pairs.loc[pairs["system"] == "random", "seconds"] > 0).all()  # kept when skipped


def test_summarize_rewrites_summary_from_disk(tmp_path, capsys):
    data = small_data()
    systems = [NullSystem(), RandomSystem(min_events=20)]
    outcome = run_experiment(small_config(), systems, tmp_path, data=data)
    summary = tmp_path / SUMMARY_DIR
    before = {name: read_summary(summary, name) for name in SUMMARY_FILES}
    for path in summary.iterdir():
        path.unlink()
    (summary / "per_universe.csv").write_text("system\nstale\n", encoding="utf-8")

    assert summarize.main(["--results", str(tmp_path)]) == 0
    assert "null, random" in capsys.readouterr().out
    for name in SUMMARY_FILES:
        pd.testing.assert_frame_equal(read_summary(summary, name), before[name])
    index = json.loads((summary / INDEX_FILE).read_text(encoding="utf-8"))
    assert index["plan_sha256"] == outcome.plan_sha256
    assert plan_sha256(load_plan(tmp_path / PLAN_FILE)) == outcome.plan_sha256


def test_summarize_ignores_runs_of_another_plan(tmp_path):
    data = small_data()
    run_experiment(small_config(), [NullSystem(), RandomSystem(min_events=20)], tmp_path,
                   data=data)
    events = data.events.iloc[1:].reset_index(drop=True)
    run_experiment(small_config(), [NullSystem()], tmp_path,
                   data=BaseData(data.prices, data.benchmarks, events))
    assert summarize.main(["--results", str(tmp_path)]) == 0
    assert set(read_summary(tmp_path / SUMMARY_DIR, "per_universe")["system"]) == {"null"}


def test_summarize_without_a_plan_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        summarize.main(["--results", str(tmp_path)])


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
    pairs = read_summary(tmp_path / SUMMARY_DIR, "pairs")
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
