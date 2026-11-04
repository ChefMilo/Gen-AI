"""Group and truth-level comparisons, FD rates and read_summary, against hand-computed values
from tests/summary_fixtures.py (each planted universe holds one feature type)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from evaluation.metrics import MetricsTables
from evaluation.stats import (
    compare_systems,
    detection_cells_data,
    group_comparison,
    group_comparisons,
    mcnemar_exact,
    paired_comparison,
    summary_frames,
    system_summary,
    truth_comparison,
    truth_comparisons,
)
from evaluation.summary import ordered_systems, read_summary
from summary_fixtures import fixture_tables, write_fixture_summary

N_BOOT = 500
TEXT = ("feature_type", "text")


@pytest.fixture(scope="module")
def tables():
    return fixture_tables()


# Group comparison: universe level


def test_group_comparison_uses_only_universes_holding_the_group(tables):
    res = group_comparison(tables, "r2", "r3", *TEXT, n_boot=N_BOOT)
    assert res.n_universes == 4
    assert list(res.differences["universe_id"]) == ["t1", "t2", "t3", "t4"]
    # r2: 3/8 text truths; r3: 6/8.
    assert res.estimate == pytest.approx(3 / 8 - 6 / 8)
    # t1 1/2 v 2/2; t2 0/2 v 2/2; t3 1/2 v 1/2; t4 1/2 v 1/2.
    np.testing.assert_allclose(res.differences["diff"], [-0.5, -1.0, 0.0, 0.0])
    assert res.test == "wilcoxon"
    assert res.p_value == pytest.approx(sp_stats.wilcoxon([-0.5, -1.0, 0.0, 0.0]).pvalue)


def test_group_comparison_structured_universes(tables):
    res = group_comparison(tables, "r2", "r3", "feature_type", "structured", n_boot=N_BOOT)
    assert res.n_universes == 4
    assert res.estimate == pytest.approx(5 / 8 - 6 / 8)  # only s4's weak truth differs
    assert list(res.differences["universe_id"]) == ["s1", "s2", "s3", "s4"]


def test_group_comparison_strength_spans_both_feature_types(tables):
    res = group_comparison(tables, "r2", "r3", "strength", "weak", n_boot=N_BOOT)
    assert res.n_universes == 8  # every universe holds one weak truth
    # weak truths: r2 finds t1, t3, t4, s1 = 4/8; r3 finds t1, t2, t4, s1, s4 = 5/8.
    assert res.estimate == pytest.approx(4 / 8 - 5 / 8)


def test_group_comparison_drops_a_universe_one_system_lacks(tables):
    truths = tables.truths
    dropped = truths[~((truths["system"] == "r3") & (truths["universe_id"] == "t2"))]
    res = group_comparison(tables._replace(truths=dropped), "r2", "r3", *TEXT, n_boot=N_BOOT)
    assert res.n_universes == 3
    assert "t2" not in set(res.differences["universe_id"])


def test_group_comparison_matches_paired_comparison_on_a_hand_built_table(tables):
    # The aggregated rows through compare_systems' own function, same draws.
    rows = []
    counts = {"t1": (1, 2), "t2": (0, 2), "t3": (1, 1), "t4": (1, 1)}
    for universe, (r2, r3) in counts.items():
        for system, matched in (("r2", r2), ("r3", r3)):
            rows.append({"system": system, "universe_id": universe, "universe_type": "planted",
                         "n_truths": 2, "n_matched": matched})
    built = MetricsTables(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
    direct = paired_comparison(built, "r2", "r3", "recall", seed=3, n_boot=N_BOOT)
    res = group_comparison(tables, "r2", "r3", *TEXT, seed=3, n_boot=N_BOOT)
    assert (res.estimate, res.lo, res.hi) == pytest.approx((direct.estimate, direct.lo, direct.hi))
    assert res.p_value == pytest.approx(direct.p_value)


def test_group_comparisons_table(tables):
    out = group_comparisons(tables, n_boot=50)
    # 3 pairs x (2 feature types + 2 strengths + 2 complexities) x 2 rules.
    assert len(out) == 3 * 6 * 2
    row = out.query("system_a == 'r2' and system_b == 'r3' and group == 'feature_type' "
                    "and value == 'text' and metric == 'recall'").iloc[0]
    assert (row["recall_a"], row["recall_b"]) == (3 / 8, 6 / 8)
    assert row["estimate"] == pytest.approx(-3 / 8)
    assert row["n_universes"] == 4


# Truth-level paired test


def test_truth_comparison_counts_and_mcnemar(tables):
    row = truth_comparison(tables, "r2", "r3", *TEXT, n_boot=N_BOOT)
    # t1 single both; t1 conj r3; t2 single r3; t2 conj r3; t3 single r2; t3 conj r3;
    # t4 single neither; t4 conj both.
    assert (row["a_only"], row["b_only"], row["both"], row["neither"]) == (1, 4, 2, 1)
    assert row["n_truths"] == 8 and row["n_universes"] == 4
    assert row["recall_a"] == pytest.approx(3 / 8)
    assert row["recall_b"] == pytest.approx(6 / 8)
    assert row["estimate"] == pytest.approx(-3 / 8)
    # Discordant 5, the rarer side 1: 2 * (C(5,0) + C(5,1)) / 2^5.
    assert row["p_value"] == pytest.approx(2 * (1 + 5) / 32)
    assert row["lo"] <= row["estimate"] <= row["hi"]
    assert row["lo_a"] <= row["recall_a"] <= row["hi_a"]


def test_truth_comparison_ci_resamples_universes_and_shares_the_group_draws(tables):
    truth = truth_comparison(tables, "r2", "r3", *TEXT, seed=5, n_boot=N_BOOT)
    universe = group_comparison(tables, "r2", "r3", *TEXT, seed=5, n_boot=N_BOOT)
    assert (truth["lo"], truth["hi"]) == pytest.approx((universe.lo, universe.hi))
    assert truth["estimate"] == pytest.approx(universe.estimate)


def test_truth_comparison_bootstrap_by_hand():
    # Two universes, one truth each: A finds u1 only, B finds u2 only.
    truths = pd.DataFrame([
        {"system": s, "universe_id": u, "spec_id": "x", "feature_type": "text",
         "matched": found, "matched_uniform": found}
        for s, u, found in [("a", "u1", True), ("a", "u2", False),
                            ("b", "u1", False), ("b", "u2", True)]
    ])
    built = MetricsTables(pd.DataFrame(), pd.DataFrame(), truths)
    row = truth_comparison(built, "a", "b", *TEXT, n_boot=2000, seed=0)
    assert row["estimate"] == 0.0
    # Replicates: {u1,u1} -> +1, {u2,u2} -> -1, mixed -> 0.
    assert (row["lo"], row["hi"]) == (-1.0, 1.0)
    assert row["p_value"] == 1.0


def test_truth_comparison_uniform_uses_matched_uniform(tables):
    row = truth_comparison(tables, "r2", "r3", *TEXT, "recall_uniform", n_boot=N_BOOT)
    # R3 loses t1's single truth under the uniform rule, which becomes R2's alone.
    assert (row["a_only"], row["b_only"], row["both"], row["neither"]) == (2, 4, 1, 1)
    assert row["recall_b"] == pytest.approx(5 / 8)


def test_truth_comparison_without_uniform_rule_is_nan(tables):
    truths = tables.truths.assign(matched_uniform=None)
    row = truth_comparison(tables._replace(truths=truths), "r2", "r3", *TEXT, "recall_uniform",
                           n_boot=10)
    assert row["n_truths"] == 8
    assert math.isnan(row["estimate"]) and math.isnan(row["p_value"])


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0


def test_truth_comparisons_table(tables):
    out = truth_comparisons(tables, n_boot=20)
    assert len(out) == 3 * 6 * 2
    assert set(out["metric"]) == {"recall", "recall_uniform"}
    counts = out[["a_only", "b_only", "both", "neither"]].sum(axis=1)
    assert (counts == out["n_truths"]).all()


# False discoveries per test


def test_false_discoveries_per_test(tables):
    out = system_summary(tables, seed=0, n_boot=N_BOOT).set_index("system")
    assert out.loc["r3", "false_discoveries_per_test"] == pytest.approx(1 / 60)
    assert out.loc["r3-u", "false_discoveries_per_test"] == pytest.approx(4 / 200)
    assert out.loc["r2", "false_discoveries_per_test"] == pytest.approx(1 / 40)
    assert out.loc["r3", "false_discoveries_per_test_uniform"] == 0.0
    assert out.loc["r3-u", "false_discoveries_per_test_uniform"] == pytest.approx(1 / 200)
    row = out.loc["r3-u"]
    assert (row["false_discoveries_per_test_lo"] <= row["false_discoveries_per_test"]
            <= row["false_discoveries_per_test_hi"])


def test_share_with_false_discovery_uniform(tables):
    out = system_summary(tables, seed=0, n_boot=N_BOOT).set_index("system")
    assert out.loc["r3-u", "share_with_false_discovery"] == 1.0
    assert out.loc["r3-u", "share_with_false_discovery_uniform"] == 0.5
    assert out.loc["r3", "share_with_false_discovery_uniform"] == 0.0


def test_false_discovery_rate_is_nan_without_backtests():
    rows = [{"system": "null", "universe_id": "p1", "universe_type": "placebo", "n_truths": 0,
             "n_matched": 0, "n_significant": 0, "false_discoveries": 0,
             "effect_error_sum": 0.0, "effect_error_abs_sum": 0.0, "backtests_used": 0}]
    built = MetricsTables(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
    row = system_summary(built, n_boot=20).iloc[0]
    assert math.isnan(row["false_discoveries_per_test"])
    assert math.isnan(row["false_discoveries_per_test_lo"])


def test_missing_backtests_column_is_nan_not_an_error():
    rows = [{"system": s, "universe_id": "p1", "universe_type": "placebo", "n_truths": 0,
             "n_matched": 0, "n_significant": 0, "false_discoveries": 1,
             "effect_error_sum": 0.0, "effect_error_abs_sum": 0.0} for s in "ab"]
    built = MetricsTables(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
    row = compare_systems(built, metrics=["false_discoveries_per_test",
                                          "share_with_false_discovery_uniform"], n_boot=10)
    assert row["estimate"].isna().all() and row["test"].isna().all()


# Detection cells


def test_detection_cells_are_per_complexity_feature_strength(tables):
    out = detection_cells_data(tables, seed=0, n_boot=50)
    row = out.query("system == 'r3' and complexity == 'single' and feature_type == 'text' "
                    "and strength == 'weak'").iloc[0]
    # t1 found, t3 not.
    assert (row["n_truths"], row["n_matched"], row["recall"]) == (2, 1, 0.5)
    assert row["recall_uniform"] == 0.0  # t1's is lost under the uniform rule
    assert len(out) == 3 * 2 * 2 * 2
    first = out[out["system"] == "r2"].head(2)
    assert list(first["complexity"]) == ["single", "single"]
    assert list(first["feature_type"]) == ["structured", "structured"]


# read_summary and summary_frames


def test_read_summary_keeps_null_as_a_system_name(tmp_path):
    frame = pd.DataFrame({"system": ["null", "None", "NA", "r0"], "recall": [0.0, 0.1, 0.2, None]})
    frame.to_csv(tmp_path / "t.csv", index=False)
    out = read_summary(tmp_path, "t")
    assert list(out["system"]) == ["null", "None", "NA", "r0"]
    assert out["recall"].isna().tolist() == [False, False, False, True]
    assert out["recall"].dtype == float
    assert pd.read_csv(tmp_path / "t.csv")["system"].isna().any()  # why it exists


def test_summary_round_trips_through_csv(tmp_path):
    write_fixture_summary(tmp_path)
    truth = read_summary(tmp_path / "summary", "truth_comparisons")
    row = truth.query("system_a == 'r2' and system_b == 'r3' and group == 'feature_type' "
                      "and value == 'text' and metric == 'recall'").iloc[0]
    assert (row["a_only"], row["b_only"]) == (1, 4)
    truths = read_summary(tmp_path / "summary", "truths")
    assert truths["matched"].dtype == bool
    assert truths["matched_uniform"].dtype == bool


def test_summary_frames_tables(tables):
    assert set(summary_frames(tables, n_boot=10)) == {
        "system_summary", "detection_curve", "detection_cells", "comparisons",
        "group_comparisons", "truth_comparisons"}


def test_ordered_systems_puts_the_experiment_first():
    assert ordered_systems(["random", "r3-u", "null", "r0", "r3"]) == [
        "r0", "r3", "r3-u", "null", "random"]
