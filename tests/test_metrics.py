import math

import pandas as pd
import pytest

from evaluation.metrics import (
    GROUPS,
    effect_errors,
    efficiency,
    evaluate,
    metrics_table,
    placebo_summary,
    recall_breakdowns,
    retention,
    retention_summary,
)
from evaluation.universe_record import placebo_record, planted_record
from evaluation_helpers import J, finding, ids, placebo_info, truth


def planted_universe(universe_id="u1"):
    """Four truths over disjoint events, differing in every breakdown label."""
    return planted_record(universe_id, [
        truth("a", ids("a", 10), strength="strong", direction=1, effect=0.04),
        truth("b", ids("b", 10), strength="weak", direction=-1, effect=0.01,
              complexity="conjunction"),
        truth("c", ids("c", 10), strength="strong", event_type="buyback", direction=1,
              effect=0.04, feature_type="text"),
        truth("d", ids("d", 10), strength="very_weak", direction=1, effect=0.005),
    ])


def planted_findings():
    return [
        finding("fa", ids("a", 10), effect_car=0.05),  # matches a, error +0.01
        finding("fb", ids("b", 10), effect_car=-0.007),  # matches b, error +0.003
        finding("fc", ids("c", 10), event_type="buyback", effect_car=-0.03),  # wrong sign
        finding("fx", ids("x", 10), effect_car=0.02),  # significant, nowhere near a truth
        finding("fy", ids("y", 10), effect_car=0.02, passed=False),  # not significant
    ]


@pytest.fixture
def results():
    return [
        evaluate("R3", planted_universe(), planted_findings(), J,
                 metadata={"tool_calls": 12, "tokens": 5000}),
        evaluate("R3", placebo_record("p1", placebo_info(1)),
                 [finding("p1a"), finding("p1b"), finding("p1c", passed=False)], J),
        evaluate("R3", placebo_record("p2", placebo_info(2)), [finding("p2a", passed=False)], J),
        evaluate("R0", planted_universe(), [], J),
    ]


def test_effect_error_arithmetic(results):
    match = results[0].match
    errors = {m.spec_id: m.effect_error for m in match.matches}
    assert errors["a"] == pytest.approx(0.05 - 0.04)
    assert errors["b"] == pytest.approx(-0.007 - (-0.01))
    e = effect_errors(match.matches)
    assert e["effect_error_mean"] == pytest.approx((0.01 + 0.003) / 2)
    assert e["effect_error_mean_abs"] == pytest.approx((0.01 + 0.003) / 2)
    # Hand-built truths have no injected_car, so both errors use direction * effect.
    assert e["nominal_effect_error_mean"] == pytest.approx(e["effect_error_mean"])
    neg = evaluate("R3", planted_record("u", [truth()]), [finding(effect_car=0.015)], J)
    assert effect_errors(neg.match.matches) == pytest.approx({
        "effect_error_mean": -0.005, "effect_error_mean_abs": 0.005,
        "nominal_effect_error_mean": -0.005, "nominal_effect_error_mean_abs": 0.005,
    })


def test_effect_error_uses_injected_car_of_matched_measurable_targets():
    # e0, e1 overlap (0.03 each); e9 is unmeasurable; x0 is not a target.
    injected = {f"e{i}": 0.02 for i in range(10)} | {"e0": 0.03, "e1": 0.03, "e9": 0.5}
    t = truth(unmeasurable=["e9"]).model_copy(update={"injected_car": injected})
    f = finding(event_ids=ids("e", 10) + ["x0"], effect_car=0.025)
    m = evaluate("R3", planted_record("u", [t]), [f], J).match.matches[0]
    true_effect = (0.03 * 2 + 0.02 * 7) / 9
    assert m.true_effect == pytest.approx(true_effect)
    assert m.effect_error == pytest.approx(0.025 - true_effect)
    assert (m.nominal_effect, m.nominal_effect_error) == pytest.approx((0.02, 0.005))


def test_effect_errors_without_matches_are_nan():
    assert all(math.isnan(v) for v in effect_errors([]).values())


def test_per_universe_table(results):
    table = metrics_table(results).per_universe
    assert len(table) == 4
    row = table[(table["system"] == "R3") & (table["universe_id"] == "u1")].iloc[0]
    assert row["n_truths"] == 4 and row["n_matched"] == 2
    assert row["recall"] == pytest.approx(0.5)
    # Significant: fa, fb, fc, fx. Matched: fa, fb.
    assert row["n_significant"] == 4
    assert row["precision"] == pytest.approx(0.5)
    assert row["n_unmatched"] == 2
    assert row["n_near_miss"] == 1
    assert math.isnan(row["false_discoveries"])
    assert row["tool_calls"] == 12 and row["tokens"] == 5000

    p1 = table[table["universe_id"] == "p1"].iloc[0]
    assert p1["false_discoveries"] == 2 and p1["universe_type"] == "placebo"
    assert math.isnan(p1["recall"]) and math.isnan(p1["precision"])

    r0 = table[table["system"] == "R0"].iloc[0]
    assert r0["recall"] == 0 and math.isnan(r0["precision"])
    assert math.isnan(r0["effect_error_mean"])


def test_placebo_summary(results):
    summary = placebo_summary(results)
    assert summary.to_dict("records") == [{
        "system": "R3", "n_universes": 2, "share_with_false_discovery": 0.5,
        "mean_false_discoveries": 1.0,
    }]


def test_recall_breakdowns_sum_to_overall(results):
    bd = metrics_table(results).breakdowns
    for _, rows in bd.groupby("system"):
        overall = rows[rows["group"] == "overall"].iloc[0]
        for group in GROUPS:
            g = rows[rows["group"] == group]
            assert g["n_truths"].sum() == overall["n_truths"]
            assert g["n_matched"].sum() == overall["n_matched"]
    r3 = bd[bd["system"] == "R3"].set_index(["group", "value"])
    assert r3.loc[("overall", "all"), "recall"] == pytest.approx(0.5)
    assert r3.loc[("strength", "strong"), "n_truths"] == 2
    assert r3.loc[("strength", "strong"), "recall"] == pytest.approx(0.5)
    assert r3.loc[("strength", "weak"), "recall"] == 1.0
    assert r3.loc[("strength", "very_weak"), "recall"] == 0.0
    assert r3.loc[("direction", "down"), "recall"] == 1.0
    assert r3.loc[("event_type", "buyback"), "recall"] == 0.0
    assert r3.loc[("feature_type", "text"), "n_truths"] == 1


def test_recall_breakdowns_pool_universes():
    rs = [
        evaluate("R3", planted_record("u1", [truth()]), [finding()], J),
        evaluate("R3", planted_record("u2", [truth()]), [], J),
    ]
    overall = recall_breakdowns(rs).query("group == 'overall'").iloc[0]
    assert (overall["n_truths"], overall["n_matched"], overall["recall"]) == (2, 1, 0.5)


def test_empty_inputs():
    tables = metrics_table([])
    assert tables.per_universe.empty and tables.breakdowns.empty
    assert placebo_summary([]).empty


def test_duplicate_system_universe_rejected(results):
    with pytest.raises(ValueError, match="same"):
        metrics_table([results[0], results[0]])


def test_efficiency_skips_absent_fields():
    assert efficiency([finding()], {}) == {}
    assert efficiency([], {"cost_usd": 0.4, "other": 1}) == {"cost_usd": 0.4}


def test_retention_from_findings_and_dict():
    ins = [finding("a", effect_car=0.02), finding("b", effect_car=-0.01),
           finding("c", effect_car=0.03), finding("d", effect_car=0.0)]
    table = retention(ins, {"a": 0.01, "b": 0.005, "d": 0.01})
    t = table.set_index("finding_id")
    assert t.loc["a", "sign_agree"] == 1.0 and t.loc["a", "ratio"] == pytest.approx(0.5)
    assert t.loc["b", "sign_agree"] == 0.0 and t.loc["b", "ratio"] == pytest.approx(-0.5)
    assert math.isnan(t.loc["c", "holdout_car"]) and math.isnan(t.loc["c", "sign_agree"])
    assert math.isnan(t.loc["d", "ratio"])
    s = retention_summary(table)
    assert (s["n_findings"], s["n_with_holdout"]) == (4, 3)
    assert s["sign_agreement"] == pytest.approx(1 / 3)
    assert s["mean_ratio"] == pytest.approx(0.0)  # mean of 0.5 and -0.5; d has no ratio


def test_retention_from_dataframes_and_dicts():
    ins = pd.DataFrame({"finding_id": ["a"], "effect_car": [0.02]})
    hold = [{"finding_id": "a", "car": 0.03}]
    table = retention(ins, hold)
    assert table.loc[0, "ratio"] == pytest.approx(1.5) and table.loc[0, "sign_agree"] == 1.0
    assert retention([], {}).empty
