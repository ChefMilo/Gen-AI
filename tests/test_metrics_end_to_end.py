"""A perfect finding on a planted universe built from fake data scores recall 1."""

import pytest

from agent.finding import EvidenceQuote, Finding
from dsl.signal import SignalSpec
from evaluation.metrics import evaluate, metrics_table
from evaluation.reference_car import event_cars
from evaluation.universe_record import measurable_target_ids, plant_universe
from universe_helpers import zero_noise
from universes.fake_events import make_fake_events
from universes.planted import PlantSpec, TextFeatureFilter, planted_text_label
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"
H = 5
MIN_EVENTS = 20  # configs/default.yaml

# Specs whose targets partly clash (same or nearby events on a ticker), so plant_universe
# drops some, and with no overlaps within any one spec on this data (checked below):
# every remaining target's reference CAR is then exactly direction * effect.
SPECS = [
    PlantSpec(spec_id="raise_up200bp", event_type="dividend",
              condition="dividend_change_pct > 10", effect=0.02, holding_days=H,
              direction=1, strength="moderate", complexity="single",
              feature_type="structured"),
    PlantSpec(spec_id="framing_growth_small_down100bp", event_type="dividend",
              condition='mcap_bucket == "small"',
              text_feature=TextFeatureFilter(feature="dividend_framing",
                                             equals=planted_text_label("dividend_framing")),
              effect=0.01, holding_days=H, direction=-1, strength="weak",
              complexity="conjunction", feature_type="text"),
    PlantSpec(spec_id="new_program_small_down400bp", event_type="buyback",
              condition='is_new_program == True and mcap_bucket == "small"', effect=0.04,
              holding_days=H, direction=-1, strength="strong", complexity="conjunction",
              feature_type="structured"),
]


@pytest.fixture(scope="module")
def universe():
    benchmarks = make_synthetic_benchmarks(START, END, seed=0)
    prices = zero_noise(make_synthetic_prices(TICKERS, START, END, seed=0), benchmarks)
    events = make_fake_events(prices, seed=0)
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
    assert all(t.overlaps == [] for t in record.truths)
    findings = [perfect_finding(t, planted_prices, benchmarks, events) for t in record.truths]
    result = evaluate("R3", record, findings)

    assert {m.spec_id for m in result.match.matches} == {s.spec_id for s in SPECS}
    assert all(m.jaccard == 1.0 for m in result.match.matches)
    tables = metrics_table([result])
    row = tables.per_universe.iloc[0]
    assert row["recall"] == 1.0 and row["precision"] == 1.0 and row["n_unmatched"] == 0
    assert row["effect_error_mean"] == pytest.approx(0, abs=1e-9)
    assert row["effect_error_mean_abs"] == pytest.approx(0, abs=1e-9)
    overall = tables.breakdowns.query("group == 'overall'").iloc[0]
    assert overall["recall"] == 1.0
