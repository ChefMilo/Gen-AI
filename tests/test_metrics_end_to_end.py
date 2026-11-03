"""A perfect finding on a planted universe built from fake data scores recall 1."""

import pytest

from agent.finding import EvidenceQuote, Finding
from dsl.signal import SignalSpec
from evaluation.metrics import evaluate, metrics_table
from evaluation.reference_car import event_cars
from evaluation.systems import OracleSystem, make_view
from evaluation.universe_record import measurable_target_ids, plant_universe
from universe_helpers import zero_noise
from universes.fake_events import make_fake_events
from universes.planted import PlantSpec, TextFeatureFilter, planted_text_label
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"
H = 5
MIN_EVENTS = 20  # configs/default.yaml

# The two original specs plus rationale_undervalued_up200bp, which clashes with
# new_program_down100bp on many targets (dropped by plant_universe) and keeps one
# overlap within the spec on this data. Its overlapping targets carry part of each
# other's effect, so their reference CAR differs from direction * effect but equals
# injected_car.
SPECS = [
    PlantSpec(spec_id="raise_up200bp", event_type="dividend",
              condition="dividend_change_pct > 10", effect=0.02, holding_days=H,
              direction=1, strength="moderate", complexity="single",
              feature_type="structured"),
    PlantSpec(spec_id="new_program_down100bp", event_type="buyback",
              condition="is_new_program == True", effect=0.01, holding_days=H,
              direction=-1, strength="weak", complexity="single", feature_type="structured"),
    PlantSpec(spec_id="rationale_undervalued_up200bp", event_type="buyback",
              text_feature=TextFeatureFilter(feature="buyback_rationale",
                                             equals=planted_text_label("buyback_rationale")),
              effect=0.02, holding_days=H, direction=1, strength="moderate",
              complexity="single", feature_type="text"),
]


@pytest.fixture(scope="module")
def data():
    benchmarks = make_synthetic_benchmarks(START, END, seed=0)
    prices = zero_noise(make_synthetic_prices(TICKERS, START, END, seed=0), benchmarks)
    return prices, benchmarks, make_fake_events(prices, seed=0)


@pytest.fixture(scope="module")
def universe(data):
    prices, benchmarks, events = data
    planted_prices, record = plant_universe("planted-0", prices, events, SPECS, MIN_EVENTS)
    return planted_prices, benchmarks, events, record


def perfect_finding(truth, planted_prices, benchmarks, events) -> Finding:
    """Measurable targets as the event set; their mean reference CAR as the effect."""
    event_ids = sorted(measurable_target_ids(truth))
    cars = event_cars(planted_prices, benchmarks, events[events["event_id"].isin(event_ids)], H)
    assert len(cars) == len(event_ids)
    effect_car = float(cars["car"].mean())
    return Finding(
        finding_id=f"f-{truth.spec.spec_id}", session_id="s-0", system="R3",
        signal_id=f"sig-{truth.spec.spec_id}",
        signal_spec=SignalSpec(event_type=truth.spec.event_type),
        event_type=truth.spec.event_type, event_ids=event_ids, n_events=len(event_ids),
        direction="long" if effect_car > 0 else "short", effect_car=effect_car,
        t_stat=10.0, p_value_raw=1e-6, p_value_corrected=1e-5, passed_correction=True,
        rationale="perfect", evidence=[EvidenceQuote(event_id=event_ids[0], quote="q")],
    )


def test_perfect_findings_recover_every_truth(universe):
    planted_prices, benchmarks, events, record = universe
    assert [t.spec.spec_id for t in record.truths] == [s.spec_id for s in SPECS]
    assert record.dropped_targets  # the specs did clash
    assert any(t.overlaps for t in record.truths)  # and one has a within-spec overlap
    findings = [perfect_finding(t, planted_prices, benchmarks, events) for t in record.truths]
    result = evaluate("R3", record, findings)

    assert {m.spec_id for m in result.match.matches} == {s.spec_id for s in SPECS}
    assert all(m.jaccard == 1.0 for m in result.match.matches)
    tables = metrics_table([result])
    row = tables.per_universe.iloc[0]
    assert row["recall"] == 1.0 and row["precision"] == 1.0 and row["n_unmatched"] == 0
    assert row["effect_error_mean"] == pytest.approx(0, abs=1e-9)
    assert row["effect_error_mean_abs"] == pytest.approx(0, abs=1e-9)
    # Against direction * effect, the overlap shows up as an error.
    assert row["nominal_effect_error_mean_abs"] > 1e-6
    overall = tables.breakdowns.query("group == 'overall'").iloc[0]
    assert overall["recall"] == 1.0


def test_perfect_finding_on_new_program_alone(data):
    """new_program_down100bp keeps its own within-spec overlap only when planted alone."""
    prices, benchmarks, events = data
    planted_prices, record = plant_universe("planted-1", prices, events, SPECS[1:2],
                                            MIN_EVENTS)
    truth = record.truths[0]
    assert truth.overlaps
    finding = perfect_finding(truth, planted_prices, benchmarks, events)
    match = evaluate("R3", record, [finding]).match.matches[0]
    assert match.effect_error == pytest.approx(0, abs=1e-9)
    assert abs(match.nominal_effect_error) > 1e-6


def test_perfect_findings_on_noisy_prices_have_zero_effect_error(data):
    """With baseline_car, the unplanted CAR of the targets no longer counts as error."""
    _, benchmarks, events = data
    noisy = make_synthetic_prices(TICKERS, START, END, seed=0)
    planted_prices, record = plant_universe("planted-2", noisy, events, SPECS, MIN_EVENTS,
                                            benchmarks=benchmarks)
    findings = [perfect_finding(t, planted_prices, benchmarks, events) for t in record.truths]
    result = evaluate("R3", record, findings)
    assert len(result.match.matches) == len(SPECS)
    for m in result.match.matches:
        assert m.effect_error == pytest.approx(0, abs=1e-12)
        assert abs(m.effect_car - m.injected_effect) > 1e-6  # the baseline it now removes
    truths = metrics_table([result]).truths
    assert truths["snr"].notna().all() and (truths["snr"] > 0).all()


def test_negative_effect_spec_gives_negative_effect_car(data):
    """effect_car is the raw mean CAR, signed as computed; direction is a separate field.

    The oracle builds its findings the way the stub systems and R0 do, so a spec planted
    with direction -1 must come back with effect_car < 0 and direction "short".
    """
    prices, benchmarks, events = data
    spec = SPECS[1]
    assert spec.direction == -1
    planted_prices, record = plant_universe("planted-neg", prices, events, [spec], MIN_EVENTS)
    view = make_view("planted-neg", planted_prices, benchmarks, events)
    run = OracleSystem(holding_days=H).run_with_truths(view, 10, 0, list(record.truths))
    [finding] = run.findings
    assert finding.effect_car < 0
    assert finding.direction == "short"
    assert evaluate("R0", record, [finding]).match.matches  # the sign test passes

    # Matching reads the sign of effect_car only: a "long" label does not flip it.
    relabelled = finding.model_copy(update={"direction": "long"})
    assert relabelled.effect_car < 0
    assert evaluate("R0", record, [relabelled]).match.matches
