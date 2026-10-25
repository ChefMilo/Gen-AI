import numpy as np
import pandas as pd
import pytest

import universes.planted as planted_module
from evaluation.reference_car import CAR_COLUMNS, event_cars, summary
from universe_helpers import zero_noise
from universes.fake_events import make_fake_events
from universes.planted import (
    GRID_DIRECTIONS,
    STRENGTHS,
    PlantSpec,
    buyback_size_threshold,
    candidate_specs,
    default_plant_min_events,
    default_spec_grid,
    eligible,
    make_planted,
    round_sig,
    select_targets,
    spillover_ids,
)
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

TICKERS = [f"T{i:02d}" for i in range(30)]
START, END = "2015-01-01", "2020-12-31"
H = 5
MIN_EVENTS = 20  # configs/default.yaml


SPEC = PlantSpec(
    spec_id="raise",
    event_type="dividend",
    condition="dividend_change_pct > 10",
    effect=0.02,
    holding_days=H,
    strength="moderate",
    complexity="single",
    feature_type="structured",
)


@pytest.fixture(scope="module")
def benchmarks():
    return make_synthetic_benchmarks(START, END, seed=0)


@pytest.fixture(scope="module")
def prices(benchmarks):
    return zero_noise(make_synthetic_prices(TICKERS, START, END, seed=0), benchmarks)


@pytest.fixture(scope="module")
def events(prices):
    return make_fake_events(prices, seed=0)


@pytest.fixture(scope="module")
def planted(prices, events):
    return make_planted(prices, events, SPEC, MIN_EVENTS)


def window_mask(prices, events, holding_days):
    """Boolean mask over prices rows inside any of the events' windows."""
    mask = pd.Series(False, index=prices.index)
    days = pd.to_datetime(events["first_tradeable_date"])
    for ticker, day0 in zip(events["ticker"], days, strict=True):
        rows = prices[prices["ticker"] == ticker].sort_values("date")
        pos = rows["date"].searchsorted(day0)
        mask[rows.index[pos : pos + holding_days]] = True
    return mask


def test_zero_noise_prices_have_zero_reference_car(prices, benchmarks, events):
    cars = event_cars(prices, benchmarks, eligible(events), H)
    assert len(cars) > 0
    np.testing.assert_allclose(cars["car"], 0.0, atol=1e-12)


def test_target_car_equals_planted_effect(planted, benchmarks, events):
    planted_prices, truth = planted
    overlapping = {e for o in truth.overlaps for e in o.event_ids}
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    cars = event_cars(planted_prices, benchmarks, targets, H)
    clean = cars[~cars["event_id"].isin(overlapping)]
    assert len(clean) >= MIN_EVENTS
    np.testing.assert_allclose(clean["car"], SPEC.effect, rtol=0, atol=1e-9)
    assert summary(clean)["hit_rate"] == 1.0


def test_non_target_car_unchanged(planted, prices, benchmarks, events):
    planted_prices, truth = planted
    others = eligible(events)
    others = others[~others["event_id"].isin(truth.target_event_ids)]
    # Windows of non-targets that touch a target window would move by design.
    touched = window_mask(prices, events[events["event_id"].isin(truth.target_event_ids)], H)
    clean = others[~others["event_id"].isin(_touching(prices, others, touched))]
    assert len(clean) > len(others) / 2
    before = event_cars(prices, benchmarks, clean, H)
    after = event_cars(planted_prices, benchmarks, clean, H)
    pd.testing.assert_frame_equal(before, after)


def _touching(prices, events, touched):
    """event_ids whose window includes a touched row."""
    ids = []
    days = pd.to_datetime(events["first_tradeable_date"])
    for event_id, ticker, day0 in zip(events["event_id"], events["ticker"], days, strict=True):
        rows = prices[prices["ticker"] == ticker].sort_values("date")
        pos = rows["date"].searchsorted(day0)
        if touched[rows.index[pos : pos + H]].any():
            ids.append(event_id)
    return ids


def test_adj_close_consistent_with_ret(planted):
    df = planted[0].sort_values(["ticker", "date"])
    expected = df.groupby("ticker")["adj_close"].pct_change()
    np.testing.assert_allclose(df["ret"], expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_rows_outside_windows_unchanged(planted, prices, events):
    planted_prices, truth = planted
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    inside = window_mask(prices, targets, H)
    assert inside.sum() == truth.n_targets * H - sum(o.n_days for o in truth.overlaps)

    # ret and every column but adj_close are identical outside the windows; benchmarks
    # are never touched (make_planted does not take them).
    pd.testing.assert_frame_equal(
        planted_prices.loc[~inside].drop(columns="adj_close"),
        prices.loc[~inside].drop(columns="adj_close"),
    )
    pd.testing.assert_frame_equal(
        planted_prices.drop(columns=["ret", "adj_close"]), prices.drop(columns=["ret", "adj_close"])
    )
    # adj_close only moves from a ticker's first window on (it carries the level shift).
    first_day = (
        pd.DataFrame({"ticker": targets["ticker"],
                      "day": pd.to_datetime(targets["first_tradeable_date"])})
        .groupby("ticker")["day"].min()
    )
    before = prices["date"] < prices["ticker"].map(first_day).fillna(pd.Timestamp.max)
    pd.testing.assert_series_equal(
        planted_prices.loc[before, "adj_close"], prices.loc[before, "adj_close"]
    )
    untouched_tickers = ~prices["ticker"].isin(first_day.index)
    pd.testing.assert_frame_equal(planted_prices[untouched_tickers], prices[untouched_tickers])


def test_targets_are_eligible_matches(planted, events):
    truth = planted[1]
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    assert truth.n_targets == len(truth.target_event_ids) == len(targets)
    assert targets["is_new_announcement"].all()
    assert (targets["extraction_status"] == "passed").all()
    assert (targets["event_type"] == "dividend").all()
    assert (targets["dividend_change_pct"] > 10).all()


def test_short_direction_lowers_car(prices, benchmarks, events):
    spec = SPEC.model_copy(update={"direction": -1})
    planted_prices, truth = make_planted(prices, events, spec, MIN_EVENTS)
    overlapping = {e for o in truth.overlaps for e in o.event_ids}
    targets = events[events["event_id"].isin(truth.target_event_ids)
                     & ~events["event_id"].isin(overlapping)]
    cars = event_cars(planted_prices, benchmarks, targets, H)
    np.testing.assert_allclose(cars["car"], -SPEC.effect, rtol=0, atol=1e-9)


def test_text_feature_spec(prices, events):
    spec = PlantSpec.model_validate({
        **SPEC.model_dump(), "spec_id": "framing", "condition": None, "feature_type": "text",
        "text_feature": {"feature": "dividend_framing", "equals": "fake_growth"},
    })
    targets = select_targets(events, spec)
    assert len(targets) > 0
    labels = targets["text_features"].map(lambda tf: tf["dividend_framing"]["label"])
    assert (labels == "fake_growth").all()


def test_text_labels_chosen_by_name_not_position(events, monkeypatch):
    def targeted():
        return {s.spec_id: list(select_targets(events, s)["event_id"])
                for s in candidate_specs(events) if s.feature_type == "text"}

    before = targeted()
    assert all(before.values())
    reordered = {f: tuple(reversed(labels)) for f, labels in planted_module.FEATURE_LABELS.items()}
    monkeypatch.setattr(planted_module, "FEATURE_LABELS", reordered)
    assert targeted() == before

    monkeypatch.setitem(planted_module.PLANTED_TEXT_LABELS, "dividend_framing", "growth")
    with pytest.raises(ValueError, match="not in"):
        candidate_specs(events)


def test_deterministic(planted, prices, events):
    again_prices, again_truth = make_planted(prices, events, SPEC, MIN_EVENTS)
    pd.testing.assert_frame_equal(planted[0], again_prices)
    assert planted[1] == again_truth


def test_refuses_too_few_targets(prices, events):
    n = len(select_targets(events, SPEC))
    with pytest.raises(ValueError, match="fewer than min_events"):
        make_planted(prices, events, SPEC, min_events=n + 1)
    make_planted(prices, events, SPEC, min_events=n)


def test_overlapping_windows_apply_both(prices, benchmarks, events):
    # Two targets on one ticker, the second starting 2 trading days after the first.
    ticker = "T00"
    days = prices[prices["ticker"] == ticker].sort_values("date")["date"].iloc[500:503:2]
    base = eligible(events)
    base = base[base["event_type"] == "dividend"].iloc[[0, 1]].copy()
    base["ticker"] = ticker
    base["first_tradeable_date"] = [d.date() for d in days]
    base["event_id"] = ["ov-1", "ov-2"]
    spec = SPEC.model_copy(update={"spec_id": "overlap", "condition": None})

    planted_prices, truth = make_planted(prices, base, spec, min_events=2)
    assert len(truth.overlaps) == 1
    overlap = truth.overlaps[0]
    assert overlap.ticker == ticker and overlap.event_ids == ("ov-1", "ov-2")
    assert overlap.n_days == H - 2

    cars = event_cars(planted_prices, benchmarks, base, H).set_index("event_id")["car"]
    per_day = SPEC.effect / H
    # Each window gets its own effect plus the other's on the shared days.
    np.testing.assert_allclose(cars, SPEC.effect + per_day * (H - 2), rtol=0, atol=1e-9)
    window = planted_prices[(planted_prices["ticker"] == ticker)].sort_values("date")
    assert np.isclose(window["ret"].iloc[502] - prices.loc[window.index[502], "ret"], 2 * per_day)


def test_event_cars_skips_unmeasurable(prices, benchmarks, events):
    rows = eligible(events).iloc[:3].copy()
    last = prices[prices["ticker"] == rows["ticker"].iloc[0]]["date"].max()
    rows.iloc[0, rows.columns.get_loc("first_tradeable_date")] = last.date()
    rows.iloc[1, rows.columns.get_loc("first_tradeable_date")] = pd.Timestamp("2015-01-03").date()
    cars = event_cars(prices, benchmarks, rows, H)
    assert list(cars.columns) == CAR_COLUMNS
    skipped = cars.attrs["skipped"]
    reasons = dict(zip(skipped["event_id"], skipped["reason"], strict=True))
    assert reasons == {rows["event_id"].iloc[0]: "past_data", rows["event_id"].iloc[1]: "no_day0"}
    assert len(cars) == 1 and (cars["n_days"] == H).all()


def test_event_cars_skips_bad_price_windows(prices, benchmarks, events):
    rows = eligible(events).iloc[:1]
    ticker = rows["ticker"].iloc[0]
    day0 = pd.Timestamp(rows["first_tradeable_date"].iloc[0])
    bad = prices.copy()
    idx = bad[(bad["ticker"] == ticker) & (bad["date"] > day0)].sort_values("date").index[1]
    bad.loc[idx, "is_bad_price"] = True
    cars = event_cars(bad, benchmarks, rows, H)
    assert cars.empty and list(cars.attrs["skipped"]["reason"]) == ["bad_price"]


def test_summary():
    cars = pd.DataFrame({"car": [0.01, 0.03, -0.01, 0.01]})
    s = summary(cars)
    assert s["n"] == 4 and np.isclose(s["mean_car"], 0.01) and s["hit_rate"] == 0.75
    assert np.isclose(s["t_stat"], 0.01 / (np.std([0.01, 0.03, -0.01, 0.01], ddof=1) / 2))
    assert summary(cars.iloc[:0])["n"] == 0


def test_candidate_grid_shape(events):
    grid = candidate_specs(events)
    assert len(grid) == 6 * 2 * len(STRENGTHS)
    assert len({s.spec_id for s in grid}) == len(grid)
    assert {s.strength for s in grid} == set(STRENGTHS.values())
    assert {(s.complexity, s.feature_type) for s in grid} == {
        ("single", "structured"), ("conjunction", "structured"),
        ("single", "text"), ("conjunction", "text"),
    }
    for s in grid:
        assert ("mcap_bucket" in (s.condition or "")) == (s.complexity == "conjunction")
        assert s.direction == GRID_DIRECTIONS[s.effect]
    assert {s.direction for s in grid} == {1, -1}


def test_round_sig():
    assert [round_sig(x) for x in (6.26, 0.0347, 15.2, 0.96)] == [6.0, 0.03, 20.0, 1.0]


def test_buyback_threshold_is_top_quartile_of_eligible(events):
    rows = eligible(events)
    values = rows.loc[rows["event_type"] == "buyback", "buyback_pct_mcap"]
    threshold = buyback_size_threshold(events)
    assert threshold == round_sig(values.quantile(0.75))
    # Ineligible rows don't move it.
    noisy = pd.concat([events, events.assign(
        event_id=events["event_id"] + "-x", buyback_pct_mcap=1e6, extraction_status="flagged"
    )])
    assert buyback_size_threshold(noisy) == threshold
    grid = candidate_specs(events)
    conditions = {s.condition for s in grid if s.spec_id.startswith("buyback_large")}
    assert conditions == {f"buyback_pct_mcap > {threshold:g}",
                          f'buyback_pct_mcap > {threshold:g} and mcap_bucket == "small"'}


def test_buyback_threshold_without_buybacks(events):
    dividends = events[events["event_type"] == "dividend"]
    assert buyback_size_threshold(dividends) is None
    specs, dropped = default_spec_grid(dividends, min_events=1)
    assert all(s.event_type == "dividend" for s in specs)
    assert {d.n_targets for d in dropped} == {0}


def test_grid_drops_specs_below_min_events(events):
    counts = {s.spec_id: len(select_targets(events, s)) for s in candidate_specs(events)}
    min_events = sorted(set(counts.values()))[len(set(counts.values())) // 2]
    specs, dropped = default_spec_grid(events, min_events)
    assert {s.spec_id for s in specs} == {k for k, n in counts.items() if n >= min_events}
    assert {d.spec_id: d.n_targets for d in dropped} == {
        k: n for k, n in counts.items() if n < min_events
    }
    assert specs and dropped

    everything, none = default_spec_grid(events, min_events=0)
    assert len(everything) == len(counts) and none == []
    nothing, all_dropped = default_spec_grid(events, min_events=max(counts.values()) + 1)
    assert nothing == [] and len(all_dropped) == len(counts)


def test_negative_grid_plant_reads_back_minus_effect(prices, benchmarks, events):
    specs, _ = default_spec_grid(events, MIN_EVENTS)
    spec = next(s for s in specs if s.direction == -1 and s.feature_type == "structured")
    planted_prices, truth = make_planted(prices, events, spec, MIN_EVENTS)
    overlapping = {e for o in truth.overlaps for e in o.event_ids}
    targets = events[events["event_id"].isin(truth.target_event_ids)
                     & ~events["event_id"].isin(overlapping)]
    cars = event_cars(planted_prices, benchmarks, targets, spec.holding_days)
    assert len(cars) > 0
    np.testing.assert_allclose(cars["car"], -spec.effect, rtol=0, atol=1e-9)
    assert summary(cars)["hit_rate"] == 0.0


@pytest.mark.slow
def test_default_grid_on_full_fake_universe(large_fake_events):
    specs, dropped = default_spec_grid(large_fake_events, MIN_EVENTS)
    for spec in specs:
        assert len(select_targets(large_fake_events, spec)) >= MIN_EVENTS
    assert all(d.n_targets < MIN_EVENTS for d in dropped)
    # Every cell of the grid survives the filter in at least one spec.
    assert {(s.complexity, s.feature_type) for s in specs} == {
        ("single", "structured"), ("conjunction", "structured"),
        ("single", "text"), ("conjunction", "text"),
    }
    assert {s.direction for s in specs} == {1, -1}


def test_injected_car_equals_reference_car_change(prices, benchmarks, events):
    # One overlap on this data, so both kinds of target are covered.
    spec = PlantSpec(
        spec_id="new_program", event_type="buyback", condition="is_new_program == True",
        effect=0.01, holding_days=H, direction=-1, strength="weak", complexity="single",
        feature_type="structured",
    )
    planted_prices, truth = make_planted(prices, events, spec, MIN_EVENTS)
    overlapping = {e for o in truth.overlaps for e in o.event_ids}
    assert overlapping
    measurable = set(truth.target_event_ids) - set(truth.unmeasurable_event_ids)

    for event_id, injected in truth.injected_car.items():
        if event_id in overlapping:
            assert abs(injected) > spec.effect
        else:
            assert injected == pytest.approx(-spec.effect, abs=1e-12)

    # Zero-noise prices: a measurable target's reference CAR is exactly what was injected.
    cars = event_cars(planted_prices, benchmarks,
                      events[events["event_id"].isin(measurable)], H)
    assert len(cars) == len(measurable)
    np.testing.assert_allclose(
        cars["car"], cars["event_id"].map(truth.injected_car), rtol=0, atol=1e-12
    )
    assert truth.injected_car_mean == pytest.approx(cars["car"].mean(), abs=1e-12)


def test_strength_grid_and_spec_ids(events):
    assert STRENGTHS == {0.0025: "very_weak", 0.005: "weak", 0.01: "moderate",
                         0.02: "strong"}
    assert GRID_DIRECTIONS == {0.0025: 1, 0.005: -1, 0.01: 1, 0.02: -1}
    ids = {s.spec_id for s in candidate_specs(events)}
    assert {i for i in ids if i.startswith("dividend_raise_") and "small" not in i} == {
        "dividend_raise_up25bp", "dividend_raise_down50bp", "dividend_raise_up100bp",
        "dividend_raise_down200bp",
    }
    assert "buyback_new_program_small_down200bp" in ids


def test_grid_min_events_defaults_to_plant_min_events(events):
    assert default_plant_min_events() == 50
    assert default_spec_grid(events) == default_spec_grid(events, 50)
    _, dropped = default_spec_grid(events)
    assert all(d.n_targets < 50 for d in dropped)


def test_no_baseline_without_benchmarks(planted):
    _, truth = planted
    assert truth.baseline_car is None and truth.baseline_car_mean is None
    assert truth.baseline_car_std is None and truth.snr is None


def test_baseline_and_snr_from_unplanted_prices(benchmarks, events):
    noisy = make_synthetic_prices(TICKERS, START, END, seed=0)  # nonzero ARs
    planted_prices, truth = make_planted(noisy, events, SPEC, MIN_EVENTS, benchmarks)
    measurable = set(truth.target_event_ids) - set(truth.unmeasurable_event_ids)
    cars = event_cars(noisy, benchmarks, events[events["event_id"].isin(measurable)], H)
    assert truth.baseline_car == pytest.approx(dict(zip(cars["event_id"], cars["car"],
                                                        strict=True)), abs=0)
    assert truth.baseline_car_mean == pytest.approx(cars["car"].mean())
    assert truth.baseline_car_std == pytest.approx(cars["car"].std(ddof=1))
    assert truth.snr == pytest.approx(
        SPEC.effect / (cars["car"].std(ddof=1) / np.sqrt(len(cars))))
    # Baseline + injected is the planted reference CAR.
    after = event_cars(planted_prices, benchmarks, events[events["event_id"].isin(measurable)],
                       H).set_index("event_id")["car"]
    for event_id, car in after.items():
        assert car == pytest.approx(truth.baseline_car[event_id]
                                    + truth.injected_car[event_id], abs=1e-12)


def test_zero_noise_baseline_has_no_snr(prices, benchmarks, events):
    _, truth = make_planted(prices, events, SPEC, MIN_EVENTS, benchmarks)
    assert truth.baseline_car_mean == pytest.approx(0, abs=1e-12)
    assert truth.baseline_car_std == pytest.approx(0, abs=1e-12)
    assert truth.snr is None


def test_spillover_is_eligible_non_targets_sharing_a_planted_day(planted, prices, events):
    _, truth = planted
    targets = events[events["event_id"].isin(truth.target_event_ids)]
    touched = window_mask(prices, targets, H)
    others = eligible(events)
    others = others[~others["event_id"].isin(truth.target_event_ids)]
    expected = set(_touching(prices, others, touched))
    assert expected  # buybacks announced with a raised dividend, at least
    assert set(truth.spillover_event_ids) == expected
    assert {"buyback"} <= set(events.set_index("event_id").loc[sorted(expected), "event_type"])
    # Ineligible events never count.
    flagged = events.assign(extraction_status="flagged")
    assert spillover_ids(prices, flagged, truth.target_event_ids, H) == []
