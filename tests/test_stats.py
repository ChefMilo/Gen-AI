import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from evaluation.metrics import TRUTH_COLUMNS, MetricsTables, evaluate, metrics_table
from evaluation.stats import (
    SNR_BIN_ORDER,
    STRENGTH_ORDER,
    PooledRatio,
    bootstrap_ci,
    detection_curve_data,
    paired_comparison,
    snr_bin,
    system_summary,
)
from evaluation.universe_record import placebo_record, planted_record
from evaluation_helpers import J, finding, ids, placebo_info, truth

N_BOOT = 2000
RECALL = PooledRatio("n_matched", "n_truths")


def planted_row(system, universe_id, n_truths, n_matched, n_significant=None, err_sum=0.0,
                err_abs_sum=None):
    return {
        "system": system, "universe_id": universe_id, "universe_type": "planted",
        "n_truths": n_truths, "n_matched": n_matched,
        "n_significant": n_matched if n_significant is None else n_significant,
        "false_discoveries": float("nan"), "effect_error_sum": err_sum,
        "effect_error_abs_sum": abs(err_sum) if err_abs_sum is None else err_abs_sum,
    }


def placebo_row(system, universe_id, false_discoveries):
    return {
        "system": system, "universe_id": universe_id, "universe_type": "placebo",
        "n_truths": 0, "n_matched": 0, "n_significant": false_discoveries,
        "false_discoveries": false_discoveries, "effect_error_sum": 0.0,
        "effect_error_abs_sum": 0.0,
    }


def tables(rows, truths=None) -> MetricsTables:
    return MetricsTables(pd.DataFrame(rows), pd.DataFrame(),
                         pd.DataFrame(truths or [], columns=TRUTH_COLUMNS))


# bootstrap_ci


def test_estimate_is_pooled_not_mean_of_ratios():
    df = pd.DataFrame([planted_row("R3", "u1", 1, 1), planted_row("R3", "u2", 3, 0)])
    ci = bootstrap_ci(df, RECALL, n_boot=N_BOOT, seed=0)
    assert ci["estimate"] == pytest.approx(0.25)  # mean of ratios would be 0.5
    assert ci["n_universes"] == 2
    # Replicates: {u1,u1} -> 1, {u2,u2} -> 0, mixed -> 0.25 (probability 1/2).
    assert (ci["lo"], ci["hi"]) == (0.0, 1.0)


def test_pooled_ratio_fast_path_equals_generic_statistic():
    rng = np.random.default_rng(1)
    df = pd.DataFrame([
        planted_row("R3", f"u{i}", int(n), int(m))
        for i, (n, m) in enumerate(zip(rng.integers(1, 6, 12), rng.integers(0, 2, 12),
                                       strict=True))
    ])
    def generic(rows):
        return rows["n_matched"].sum() / rows["n_truths"].sum()

    fast = bootstrap_ci(df, RECALL, n_boot=500, seed=7)
    slow = bootstrap_ci(df, generic, n_boot=500, seed=7)
    assert fast == pytest.approx(slow)
    assert fast["lo"] < fast["estimate"] < fast["hi"]


def test_resamples_whole_universes():
    # Two rows per universe: every replicate must hold both rows of each drawn universe.
    df = pd.DataFrame({"universe_id": ["a", "a", "b", "b", "c", "c"], "x": range(6)})

    def pairs_kept(rows):
        assert (rows.groupby("universe_id").size() % 2 == 0).all()
        return float(len(rows))

    ci = bootstrap_ci(df, pairs_kept, n_boot=200, seed=0)
    assert ci == {"estimate": 6.0, "lo": 6.0, "hi": 6.0, "n_universes": 3}


def test_seed_reproducible_and_alpha_widens():
    df = pd.DataFrame([planted_row("R3", f"u{i}", 4, i % 4) for i in range(10)])
    assert bootstrap_ci(df, RECALL, n_boot=N_BOOT, seed=3) == bootstrap_ci(
        df, RECALL, n_boot=N_BOOT, seed=3)
    narrow = bootstrap_ci(df, RECALL, n_boot=N_BOOT, alpha=0.5, seed=3)
    wide = bootstrap_ci(df, RECALL, n_boot=N_BOOT, alpha=0.01, seed=3)
    assert wide["lo"] <= narrow["lo"] <= narrow["hi"] <= wide["hi"]


def test_zero_denominator_replicates_left_out():
    # u2 has no truths: drawing only u2 gives 0/0, which is not a recall.
    df = pd.DataFrame([planted_row("R3", "u1", 2, 1), planted_row("R3", "u2", 0, 0)])
    ci = bootstrap_ci(df, RECALL, n_boot=N_BOOT, seed=0)
    assert (ci["estimate"], ci["lo"], ci["hi"]) == (0.5, 0.5, 0.5)


def test_empty_input():
    ci = bootstrap_ci(pd.DataFrame(columns=["universe_id"]), RECALL)
    assert ci["n_universes"] == 0 and math.isnan(ci["estimate"]) and math.isnan(ci["lo"])


# system_summary


def summary_rows():
    return [
        planted_row("R3", "u1", 4, 3, n_significant=4, err_sum=0.003, err_abs_sum=0.005),
        planted_row("R3", "u2", 2, 0, n_significant=1),
        planted_row("R3", "u3", 2, 1, n_significant=1, err_sum=-0.001),
        placebo_row("R3", "p1", 0),
        placebo_row("R3", "p2", 3),
        placebo_row("R3", "p3", 1),
        placebo_row("R3", "p4", 0),
        {**planted_row("R3", "real", 0, 0), "universe_type": "real"},
        planted_row("R0", "u1", 4, 1),
    ]


def test_system_summary_pooled_metrics():
    out = system_summary(tables(summary_rows()), seed=0, n_boot=N_BOOT)
    assert list(out["system"]) == ["R3", "R0"]
    r3 = out.iloc[0]
    assert (r3["n_planted"], r3["n_placebo"], r3["n_real"]) == (3, 4, 1)
    assert r3["recall"] == pytest.approx(4 / 8)
    assert r3["precision"] == pytest.approx(4 / 6)
    assert r3["effect_error_mean"] == pytest.approx((0.003 - 0.001) / 4)
    assert r3["effect_error_mean_abs"] == pytest.approx((0.005 + 0.001) / 4)
    assert r3["mean_false_discoveries"] == pytest.approx(1.0)
    assert r3["share_with_false_discovery"] == pytest.approx(0.5)
    for metric in ["recall", "precision", "mean_false_discoveries"]:
        assert r3[f"{metric}_lo"] <= r3[metric] <= r3[f"{metric}_hi"]
    r0 = out.iloc[1]
    assert (r0["n_planted"], r0["n_placebo"]) == (1, 0)
    assert r0["recall"] == pytest.approx(0.25)
    assert math.isnan(r0["mean_false_discoveries"])


def test_system_summary_selected_systems():
    out = system_summary(tables(summary_rows()), systems=["R0"], seed=0, n_boot=100)
    assert list(out["system"]) == ["R0"]


# detection_curve_data


def truth_row(system, universe_id, spec_id, strength, matched, feature_type="structured",
              complexity="single"):
    return {"system": system, "universe_id": universe_id, "spec_id": spec_id,
            "strength": strength, "complexity": complexity, "feature_type": feature_type,
            "event_type": "dividend", "direction": "up", "matched": matched}


def test_detection_curve_data():
    rows = [planted_row("R3", "u1", 3, 2), planted_row("R3", "u2", 2, 1)]
    truths = [
        truth_row("R3", "u1", "a", "strong", True),
        truth_row("R3", "u1", "b", "weak", True, feature_type="text"),
        truth_row("R3", "u1", "c", "very_weak", False, complexity="conjunction"),
        truth_row("R3", "u2", "a", "strong", True),
        truth_row("R3", "u2", "d", "moderate", False, feature_type="text"),
    ]
    out = detection_curve_data(tables(rows, truths), seed=0, n_boot=N_BOOT)
    strength = out[out["group"] == "strength"]
    assert list(strength["value"]) == STRENGTH_ORDER[:4]  # no very_strong truths here
    assert list(strength["recall"]) == [0.0, 1.0, 0.0, 1.0]
    assert list(out["group"].unique()) == ["strength", "feature_type", "complexity"]
    # Each group's counts add up to all truths.
    for _, g in out.groupby("group"):
        assert g["n_truths"].sum() == 5 and g["n_matched"].sum() == 3
    text = out[(out["group"] == "feature_type") & (out["value"] == "text")].iloc[0]
    assert (text["n_truths"], text["recall"], text["n_universes"]) == (2, 0.5, 2)
    # weak truths only exist in u1: drawing only u2 gives no weak truth, so 1.0 throughout.
    weak = strength[strength["value"] == "weak"].iloc[0]
    assert (weak["lo"], weak["hi"]) == (1.0, 1.0)


def test_snr_bin_edges():
    values = [0.0, 0.99, 1.0, 1.99, 2.0, 3.0, 3.99, 4.0, 40.0]
    assert [snr_bin(v) for v in values] == [
        "<1", "<1", "1-2", "1-2", "2-3", "3-4", "3-4", ">=4", ">=4"]
    assert snr_bin(float("nan")) is None


def test_detection_curve_by_snr_bin():
    rows = [planted_row("R3", "u1", 3, 2), planted_row("R3", "u2", 3, 1)]
    truths = [
        {**truth_row("R3", "u1", "a", "strong", True), "snr": 5.0},
        {**truth_row("R3", "u1", "b", "weak", True), "snr": 1.5},
        {**truth_row("R3", "u1", "c", "very_weak", False), "snr": 0.4},
        {**truth_row("R3", "u2", "a", "strong", True), "snr": 4.0},
        {**truth_row("R3", "u2", "b", "weak", False), "snr": 1.0},
        {**truth_row("R3", "u2", "c", "very_weak", False), "snr": float("nan")},
    ]
    out = detection_curve_data(tables(rows, truths), seed=0, n_boot=N_BOOT)
    snr = out[out["group"] == "snr_bin"].set_index("value")
    assert list(snr.index) == [v for v in SNR_BIN_ORDER if v in {"<1", "1-2", ">=4"}]
    assert snr["n_truths"].to_dict() == {"<1": 1, "1-2": 2, ">=4": 2}  # NaN left out
    assert snr["recall"].to_dict() == {"<1": 0.0, "1-2": 0.5, ">=4": 1.0}
    assert list(out["group"].unique()) == ["strength", "feature_type", "complexity",
                                           "snr_bin"]


# paired_comparison


def paired_rows(a_matched, b_matched, n_truths=4):
    return (
        [planted_row("A", f"u{i}", n_truths, m) for i, m in enumerate(a_matched)]
        + [planted_row("B", f"u{i}", n_truths, m) for i, m in enumerate(b_matched)]
    )


def test_paired_requires_same_universes():
    rows = paired_rows([1, 2, 3], [1, 2]) + [planted_row("B", "u9", 4, 1)]
    with pytest.raises(ValueError, match=r"only A: \['u2'\]; only B: \['u9'\]"):
        paired_comparison(tables(rows), "A", "B", "recall")


def test_paired_requires_same_universe_types():
    rows = paired_rows([1], [1])
    rows[1] = {**rows[1], "universe_type": "placebo"}
    with pytest.raises(ValueError, match="types differ"):
        paired_comparison(tables(rows), "A", "B", "recall")


def test_paired_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        paired_comparison(tables(paired_rows([1], [1])), "A", "B", "f1")


def test_paired_identical_systems_use_sign_test():
    res = paired_comparison(tables(paired_rows([1, 2, 3], [1, 2, 3])), "A", "B", "recall",
                            n_boot=N_BOOT)
    assert (res.estimate, res.lo, res.hi) == (0.0, 0.0, 0.0)
    assert (res.test, res.p_value, res.n_pairs) == ("sign", 1.0, 3)


def test_paired_difference_and_wilcoxon():
    a = [4, 3, 4, 2, 3, 4, 3, 4]
    b = [1, 3, 2, 0, 2, 1, 3, 2]
    res = paired_comparison(tables(paired_rows(a, b)), "A", "B", "recall", seed=1,
                            n_boot=N_BOOT)
    assert res.estimate == pytest.approx((sum(a) - sum(b)) / (4 * len(a)))
    diffs = (np.array(a) - np.array(b)) / 4
    np.testing.assert_allclose(res.differences["diff"], diffs)
    assert res.test == "wilcoxon"
    assert res.p_value == pytest.approx(sp_stats.wilcoxon(diffs).pvalue)
    # A >= B in every universe with equal truths, so every paired replicate is >= 0;
    # resampling the systems independently would not guarantee this.
    assert 0.0 <= res.lo <= res.estimate <= res.hi
    assert res.n_universes == len(a)


def test_paired_skips_undefined_universes_in_test_only():
    # universe_id: (A matched, A significant), (B matched, B significant)
    counts = {"u1": ((2, 2), (1, 2)), "u2": ((0, 0), (0, 1)), "u3": ((1, 1), (1, 2))}
    rows = [
        planted_row(system, u, 4, m, n_significant=sig)
        for u, pair in counts.items()
        for system, (m, sig) in zip("AB", pair, strict=True)
    ]
    res = paired_comparison(tables(rows), "A", "B", "precision", n_boot=N_BOOT)
    assert res.n_pairs == 2  # u2: A has no significant findings
    assert res.estimate == pytest.approx(3 / 3 - 2 / 5)


def test_paired_placebo_metric_uses_placebo_universes():
    rows = [placebo_row("A", "p1", 2), placebo_row("B", "p1", 0),
            placebo_row("A", "p2", 1), placebo_row("B", "p2", 1),
            planted_row("A", "u1", 4, 4), planted_row("B", "u1", 4, 0)]
    res = paired_comparison(tables(rows), "A", "B", "mean_false_discoveries", n_boot=N_BOOT)
    assert res.n_universes == 2
    assert res.estimate == pytest.approx(1.0)
    assert list(res.differences["universe_id"]) == ["p1", "p2"]


# From evaluate() through metrics_table to stats


def test_stats_on_metrics_table_output():
    record = planted_record("u1", [truth("a", ids("a", 10), strength="strong"),
                                   truth("b", ids("b", 10), strength="weak")])
    results = [
        evaluate("R3", record, [finding("f", ids("a", 10), effect_car=0.03)], J),
        evaluate("R0", record, [], J),
        evaluate("R3", placebo_record("p1", placebo_info(1)), [finding("p")], J),
        evaluate("R0", placebo_record("p1", placebo_info(1)), [], J),
    ]
    t = metrics_table(results)
    summary = system_summary(t, seed=0, n_boot=200).set_index("system")
    assert summary.loc["R3", "recall"] == 0.5 and summary.loc["R0", "recall"] == 0.0
    assert summary.loc["R3", "effect_error_mean"] == pytest.approx(0.01)
    assert summary.loc["R3", "mean_false_discoveries"] == 1.0
    curve = detection_curve_data(t, seed=0, n_boot=200)
    assert list(curve[curve["group"] == "strength"]["value"]) == ["weak", "strong"] * 2
    res = paired_comparison(t, "R3", "R0", "recall", n_boot=200)
    assert res.estimate == 0.5
