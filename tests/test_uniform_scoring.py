"""The secondary scoring rule: a uniform Holm re-correction over backtest_budget
(evaluation.metrics.recorrect), reported next to each system's own passed_correction."""

import math

import pytest

from agent.finding import EvidenceQuote, Finding
from dsl.signal import SignalSpec
from evaluation.metrics import (
    UniformCorrection,
    evaluate,
    metrics_table,
    recorrect,
    uniform_correction,
)
from evaluation.stats import system_summary
from evaluation.universe_record import planted_record
from universes.planted import PlantSpec, PlantTruth

RULE = UniformCorrection(budget=30, alpha=0.05)


def finding(fid: str, p_raw: float, passed: bool, ids=("e1", "e2", "e3"),
            effect=0.02) -> Finding:
    return Finding(
        finding_id=fid, session_id="s", system="R0", signal_id=fid,
        signal_spec=SignalSpec(event_type="dividend"), event_type="dividend",
        event_ids=list(ids), n_events=len(ids), direction="long", effect_car=effect,
        t_stat=3.0, p_value_raw=p_raw, p_value_corrected=p_raw, passed_correction=passed,
        rationale="r", evidence=[EvidenceQuote(event_id=ids[0], quote="q")],
    )


def record():
    spec = PlantSpec(spec_id="a", event_type="dividend", condition="x > 1", effect=0.02,
                     strength="strong", complexity="single", feature_type="structured")
    truth = PlantTruth(spec=spec, target_event_ids=["e1", "e2", "e3"], n_targets=3,
                       overlaps=[], unmeasurable_event_ids=[])
    return planted_record("u1", [truth])


def test_uniform_rule_comes_from_default_config():
    rule = uniform_correction()
    assert rule.alpha == 0.05 and rule.budget == 30
    assert uniform_correction(budget=7).budget == 7


def test_recorrect_is_holm_over_the_budget():
    out = recorrect([finding("a", 0.001, False), finding("b", 0.01, True),
                     finding("c", math.nan, True)], RULE)
    assert [f.p_value_corrected for f in out] == pytest.approx([0.03, 0.29, 1.0])
    assert [f.passed_correction for f in out] == [True, False, False]
    # More findings than the budget: the family grows to cover them.
    many = recorrect([finding(str(i), 0.001, True) for i in range(40)], RULE)
    assert many[0].p_value_corrected == pytest.approx(0.04)
    assert recorrect([], RULE) == []


def test_metrics_report_both_rules():
    # The system passed it under its own (lenient) family; uniformly it does not.
    lenient = evaluate("R0", record(), [finding("a", 0.01, True)], 0.6, uniform=RULE)
    # Not passed by the system, but the uniform rule would pass it.
    strict = evaluate("R1", record(), [finding("a", 0.0001, False)], 0.6, uniform=RULE)
    tables = metrics_table([lenient, strict])
    rows = tables.per_universe.set_index("system")
    assert rows.loc["R0", "recall"] == 1.0 and rows.loc["R0", "recall_uniform"] == 0.0
    assert rows.loc["R1", "recall"] == 0.0 and rows.loc["R1", "recall_uniform"] == 1.0
    assert list(tables.truths["matched_uniform"]) == [False, True]
    overall = tables.breakdowns.query("group == 'overall'").set_index("system")
    assert overall.loc["R0", "recall_uniform"] == 0.0
    summary = system_summary(tables, n_boot=20).set_index("system")
    assert summary.loc["R0", "recall"] == 1.0 and summary.loc["R0", "recall_uniform"] == 0.0


def test_without_the_rule_uniform_columns_are_nan():
    tables = metrics_table([evaluate("R0", record(), [finding("a", 0.01, True)], 0.6)])
    assert math.isnan(tables.per_universe.loc[0, "recall_uniform"])
    summary = system_summary(tables, n_boot=20)
    assert math.isnan(summary.loc[0, "recall_uniform"])
    assert summary.loc[0, "recall"] == 1.0
