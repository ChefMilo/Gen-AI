"""evaluation.real_holdout: each system's significant real-data findings, re-tested once."""

from evaluation import real_holdout
from evaluation.run import FINDINGS, run_dir
from harness.holdout import HoldoutConfig
from test_backtest_engine import sample_data as sample_data
from test_harness import harness_data as harness_data
from test_holdout import discovery

REAL = "real-explore-v1-u00"
CONFIG = HoldoutConfig(research_end="2019-12-31", holdout_start="2020-01-06",
                       holdout_end="2020-01-13")


def write(out, system, findings):
    path = run_dir(out, system, REAL)
    path.mkdir(parents=True)
    (path / FINDINGS).write_text("".join(f.model_dump_json() + "\n" for f in findings),
                                 encoding="utf-8")


def test_significant_findings_are_tested_once(harness_data, tmp_path):
    findings = discovery(harness_data, tmp_path / "d")
    assert findings and all(f.passed_correction for f in findings)
    write(tmp_path, "r2", findings)
    calls = []

    def loader(start, end):
        calls.append((start, end))
        return harness_data

    report = real_holdout.check_system(tmp_path, "r2", REAL, CONFIG, loader)
    assert report["status"] == "completed" and report["system"] == "r2"
    assert (tmp_path / real_holdout.HOLDOUT_DIR / "r2.json").is_file()
    again = real_holdout.check_system(tmp_path, "r2", REAL, CONFIG, loader)
    assert again["note"].startswith("already run") and len(calls) == 1
    df = real_holdout.table([report])
    assert list(df["system"]) == ["r2"] and df["passed_correction"].all()


def test_systems_without_significant_findings_are_skipped(harness_data, tmp_path):
    findings = discovery(harness_data, tmp_path / "d")
    write(tmp_path, "r0", [f.model_copy(update={"passed_correction": False}) for f in findings])

    def loader(start, end):
        raise AssertionError("loaded the holdout")

    report = real_holdout.check_system(tmp_path, "r0", REAL, CONFIG, loader)
    assert report["status"] == "no significant findings to test"
    assert real_holdout.check_system(tmp_path, "r9", REAL, CONFIG, loader)["status"] == "not run"
    assert not (tmp_path / real_holdout.HOLDOUT_DIR).exists()
