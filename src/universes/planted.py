"""Planted universes: prices with a known abnormal return added after chosen events.

A PlantSpec picks target events (eligible events matching a structured condition
and/or a text feature) and an effect. make_planted adds direction * effect /
holding_days to `ret` on each of the holding_days trading days starting at the
event's first_tradeable_date, then rebuilds adj_close from the modified ret. Under
the reference event study (evaluation.reference_car), each target's CAR rises by
direction * effect and nothing else moves, except where two targets' windows overlap:
each then also carries part of the other's effect. The truth records which events
were planted, for matching findings against planted signals, and the CAR actually
injected into each target's own window (injected_car), overlaps included.

Only `ret` and `adj_close` change. close, OHLC, raw_close and mcap are left as they
were, so the events (whose mcap fields come from the prices) stay valid.

Given benchmarks, the truth also records the reference CAR of the measurable targets
on the unplanted prices (baseline_car), so that a finding's expected effect is
baseline + injected, and the plant's signal-to-noise ratio (snr). spillover_event_ids
lists eligible non-target events whose window shares a planted day: they carry part
of the effect although no truth claims them.
"""

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from store.schema import EventType
from universes.fake_events import BUYBACK_RATIONALE_LABELS, DIVIDEND_FRAMING_LABELS

DEFAULT_HOLDING_DAYS = 5
STRENGTHS = {0.0025: "very_weak", 0.005: "weak", 0.01: "moderate", 0.02: "strong"}
GRID_DIRECTIONS = {0.0025: 1, 0.005: -1, 0.01: 1, 0.02: -1}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

# Text feature -> the label the text plants target, by name (never by position in the
# label tuples, which may be reordered when Retrieval's final labels land).
FEATURE_LABELS = {
    "dividend_framing": DIVIDEND_FRAMING_LABELS,
    "buyback_rationale": BUYBACK_RATIONALE_LABELS,
}
PLANTED_TEXT_LABELS = {
    "dividend_framing": "fake_growth",
    "buyback_rationale": "fake_undervalued",
}

Strength = Literal["very_weak", "weak", "moderate", "strong", "very_strong"]
Complexity = Literal["single", "conjunction"]
FeatureType = Literal["structured", "text"]


class TextFeatureFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    equals: str


class PlantSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_id: str
    event_type: EventType
    condition: str | None = None  # pandas query over event columns
    text_feature: TextFeatureFilter | None = None
    effect: float = Field(gt=0)  # CAR added over the window, e.g. 0.02
    holding_days: int = Field(DEFAULT_HOLDING_DAYS, ge=1)
    direction: Literal[1, -1] = 1
    seed: int = 0  # the universe draw this spec belongs to; planting itself is deterministic

    # Labels for breaking down recovery rates
    strength: Strength
    complexity: Complexity
    feature_type: FeatureType


class Overlap(BaseModel):
    ticker: str
    event_ids: tuple[str, str]
    n_days: int  # trading days the two windows share


class PlantTruth(BaseModel):
    spec: PlantSpec
    target_event_ids: list[str]
    n_targets: int
    overlaps: list[Overlap]
    # Targets the reference event study cannot measure (window past the data or on a
    # bad price day). They are still planted and still count as targets.
    unmeasurable_event_ids: list[str]
    # Target event_id -> planted AR summed over its own holding window, including any
    # part from an overlapping target: direction * effect when nothing overlaps. Targets
    # with no prices row on day 0 have no window and are left out.
    injected_car: dict[str, float] = {}
    injected_car_mean: float | None = None  # over measurable targets; None if there are none
    # Reference CAR of the measurable targets on the unplanted prices, when benchmarks
    # were given (None otherwise). Targets the event study skips are left out.
    baseline_car: dict[str, float] | None = None
    baseline_car_mean: float | None = None
    baseline_car_std: float | None = None  # sample std; None below 2 CARs
    # effect / (baseline_car_std / sqrt(CARs in baseline_car)); None without a positive std.
    snr: float | None = None
    # Eligible non-target events (any type) whose window, of spec.holding_days rows,
    # shares a planted day with a target on the same ticker.
    spillover_event_ids: list[str] | None = None


def default_plant_min_events(path: Path = DEFAULT_CONFIG_PATH) -> int:
    """plant_min_events from the config: the fewest targets a planted spec may have."""
    with open(path, encoding="utf-8") as f:
        return int(yaml.safe_load(f)["plant_min_events"])


def injected_cars(
    original_ret: np.ndarray,
    planted_ret: np.ndarray,
    windows: dict[str, np.ndarray],
    measurable: set[str],
) -> tuple[dict[str, float], float | None]:
    """(event_id -> sum of planted_ret - original_ret over its window, mean over measurable)."""
    added = np.asarray(planted_ret, dtype=float) - np.asarray(original_ret, dtype=float)
    added = np.nan_to_num(added, nan=0.0)  # a NaN ret stays NaN when planted: nothing added
    injected = {event_id: float(added[rows].sum()) for event_id, rows in windows.items()}
    values = [injected[e] for e in injected if e in measurable]
    return injected, (float(np.mean(values)) if values else None)


def window_positions(
    prices: pd.DataFrame, targets: pd.DataFrame, holding_days: int
) -> dict[str, np.ndarray]:
    """event_id -> positional rows of prices in its holding window, cut short at the end
    of the data. Events with no prices row on day 0 are left out, as in make_planted."""
    dates = pd.to_datetime(prices["date"]).dt.normalize().to_numpy()
    ticker_rows = prices.groupby("ticker", sort=False).indices
    day0s = pd.to_datetime(targets["first_tradeable_date"]).dt.normalize().to_numpy()
    out = {}
    for event_id, ticker, day0 in zip(targets["event_id"], targets["ticker"], day0s,
                                      strict=True):
        rows = ticker_rows.get(ticker, np.array([], dtype=int))
        rows = rows[np.argsort(dates[rows], kind="stable")]
        pos = int(np.searchsorted(dates[rows], day0))
        if pos < len(rows) and dates[rows[pos]] == day0:
            out[event_id] = rows[pos : pos + holding_days]
    return out


def _day0_positions(days: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    """Position of each event's first_tradeable_date in days (sorted); -1 if absent."""
    day0 = pd.to_datetime(events["first_tradeable_date"]).dt.normalize().to_numpy()
    pos = np.searchsorted(days, day0)
    found = pos < len(days)
    found[found] = days[pos[found]] == day0[found]
    return np.where(found, pos, -1)


def spillover_ids(
    prices: pd.DataFrame, events: pd.DataFrame, target_ids, holding_days: int
) -> list[str]:
    """Eligible non-target events whose window shares a day with a target's window.

    Windows are the holding_days rows from first_tradeable_date on the ticker, cut
    short at the end of the data, as in make_planted. In events order.
    """
    rows = eligible(events)
    is_target = rows["event_id"].isin(set(target_ids)).to_numpy()
    targets, others = rows[is_target], rows[~is_target]
    others = others[others["ticker"].isin(set(targets["ticker"]))]
    if others.empty:
        return []
    dates = pd.to_datetime(prices["date"]).dt.normalize().to_numpy()
    ticker_rows = prices.groupby("ticker", sort=False).indices
    spill = set()
    for ticker, group in others.groupby("ticker", sort=False):
        days = np.sort(dates[ticker_rows[ticker]])
        starts = _day0_positions(days, targets[targets["ticker"] == ticker])
        starts = starts[starts >= 0]
        # planted[i]: row i is in some target's window; covered[i]: planted rows before i.
        change = np.zeros(len(days) + 1, dtype=int)
        np.add.at(change, starts, 1)
        np.add.at(change, np.minimum(starts + holding_days, len(days)), -1)
        planted = np.cumsum(change[:-1]) > 0
        covered = np.concatenate([[0], np.cumsum(planted)])
        pos = _day0_positions(days, group)
        end = np.minimum(pos + holding_days, len(days))
        hit = (pos >= 0) & (covered[end] > covered[np.maximum(pos, 0)])
        spill.update(group.loc[hit, "event_id"])
    return [e for e in events["event_id"] if e in spill]


def baseline_fields(
    prices: pd.DataFrame, benchmarks: pd.DataFrame, events: pd.DataFrame, truth: PlantTruth
) -> dict:
    """truth's baseline_car, its mean and std, and snr, on prices (unplanted)."""
    from evaluation.reference_car import event_cars  # evaluation imports this module

    measurable = set(truth.target_event_ids) - set(truth.unmeasurable_event_ids)
    cars = event_cars(prices, benchmarks, events[events["event_id"].isin(measurable)],
                      truth.spec.holding_days)
    baseline = dict(zip(cars["event_id"], cars["car"].astype(float), strict=True))
    values = np.array(list(baseline.values()), dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    snr = None
    if std is not None and std > 0:
        snr = truth.spec.effect / (std / math.sqrt(len(values)))
    return {
        "baseline_car": baseline,
        "baseline_car_mean": float(values.mean()) if len(values) else None,
        "baseline_car_std": std,
        "snr": snr,
    }


def eligible(events: pd.DataFrame) -> pd.DataFrame:
    """Events a backtest may use (docs/data_schema.md): new announcements that passed."""
    return events[events["is_new_announcement"] & (events["extraction_status"] == "passed")]


def select_targets(events: pd.DataFrame, spec: PlantSpec) -> pd.DataFrame:
    rows = eligible(events)
    rows = rows[rows["event_type"] == spec.event_type]
    if spec.condition:
        rows = rows.query(spec.condition)
    if spec.text_feature:
        feature, equals = spec.text_feature.feature, spec.text_feature.equals
        labels = rows["text_features"].map(lambda tf: (tf or {}).get(feature, {}).get("label"))
        rows = rows[labels == equals]
    return rows


def make_planted(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    spec: PlantSpec,
    min_events: int,
    benchmarks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, PlantTruth]:
    """(prices with the effect planted, truth). Raises if fewer than min_events targets.

    With benchmarks, the truth's baseline fields and snr are measured on prices.
    """
    targets = select_targets(events, spec)
    if len(targets) < min_events:
        raise ValueError(
            f"{spec.spec_id}: {len(targets)} target events, fewer than min_events={min_events}"
        )

    planted = prices.copy()
    ret = planted["ret"].to_numpy(dtype=float, copy=True)
    adj_close = planted["adj_close"].to_numpy(dtype=float, copy=True)
    dates = pd.to_datetime(planted["date"]).dt.normalize().to_numpy()
    is_bad = planted["is_bad_price"].to_numpy(dtype=bool)
    daily = spec.direction * spec.effect / spec.holding_days

    overlaps, unmeasurable = [], []
    target_windows: dict[str, np.ndarray] = {}
    day0s = pd.to_datetime(targets["first_tradeable_date"]).dt.normalize()
    target_days = pd.DataFrame(
        {"event_id": targets["event_id"], "ticker": targets["ticker"], "day0": day0s}
    )
    ticker_rows = planted.groupby("ticker", sort=False).indices
    for ticker, group in target_days.groupby("ticker", sort=False):
        rows = ticker_rows.get(ticker, np.array([], dtype=int))
        rows = rows[np.argsort(dates[rows], kind="stable")]  # positions, in date order
        windows = []
        for event_id, day0 in zip(group["event_id"], group["day0"], strict=True):
            pos = int(np.searchsorted(dates[rows], day0.to_datetime64()))
            if pos >= len(rows) or dates[rows[pos]] != day0.to_datetime64():
                unmeasurable.append(event_id)  # no prices row on day 0: nothing to plant
                continue
            window = rows[pos : pos + spec.holding_days]
            ret[window] += daily
            target_windows[event_id] = window
            if len(window) < spec.holding_days or is_bad[window].any():
                unmeasurable.append(event_id)
            windows.append((event_id, pos, pos + len(window)))
        if not windows:
            continue

        windows.sort(key=lambda w: w[1])
        for i, (a, _, end_a) in enumerate(windows):
            for b, start_b, end_b in windows[i + 1 :]:
                if start_b >= end_a:
                    break
                overlaps.append(
                    Overlap(ticker=ticker, event_ids=(a, b), n_days=min(end_a, end_b) - start_b)
                )

        # Rebuild adj_close from the first planted day, anchored on the unchanged day
        # before it (the same as anchoring at the ticker's first date).
        first = max(min(w[1] for w in windows), 1)
        anchor = adj_close[rows[first - 1]]
        adj_close[rows[first:]] = anchor * np.cumprod(1 + ret[rows[first:]])

    planted["ret"] = ret
    planted["adj_close"] = adj_close
    injected, injected_mean = injected_cars(
        prices["ret"].to_numpy(dtype=float), ret, target_windows,
        set(targets["event_id"]) - set(unmeasurable),
    )
    truth = PlantTruth(
        spec=spec,
        target_event_ids=list(targets["event_id"]),
        n_targets=len(targets),
        overlaps=overlaps,
        unmeasurable_event_ids=unmeasurable,
        injected_car=injected,
        injected_car_mean=injected_mean,
        spillover_event_ids=spillover_ids(prices, events, targets["event_id"],
                                          spec.holding_days),
    )
    if benchmarks is not None:
        truth = truth.model_copy(update=baseline_fields(prices, benchmarks, events, truth))
    return planted, truth


class DroppedSpec(BaseModel):
    spec_id: str
    n_targets: int


def round_sig(x: float, digits: int = 1) -> float:
    """x rounded to `digits` significant figures, e.g. 6.26 -> 6.0, 0.0347 -> 0.03."""
    return float(f"{x:.{digits - 1}e}")


def buyback_size_threshold(events: pd.DataFrame) -> float | None:
    """Top quartile of buyback_pct_mcap over eligible buybacks, to 1 significant figure."""
    rows = eligible(events)
    values = rows.loc[rows["event_type"] == "buyback", "buyback_pct_mcap"].dropna()
    return round_sig(float(values.quantile(0.75))) if len(values) else None


def planted_text_label(feature: str) -> str:
    """The label text plants target for feature; raises if it is not an allowed label."""
    label = PLANTED_TEXT_LABELS[feature]
    if label not in FEATURE_LABELS[feature]:
        raise ValueError(f"{feature}: planted label {label!r} not in {FEATURE_LABELS[feature]}")
    return label


def _base_specs(
    events: pd.DataFrame,
) -> list[tuple[str, EventType, str | None, TextFeatureFilter | None, FeatureType]]:
    """(name, event_type, condition, text feature, feature type) for single-feature plants."""
    framing = planted_text_label("dividend_framing")
    rationale = planted_text_label("buyback_rationale")
    threshold = buyback_size_threshold(events)
    # No eligible buybacks: the condition can match nothing, so the spec is dropped.
    large = f"buyback_pct_mcap > {'inf' if threshold is None else f'{threshold:g}'}"
    return [
        ("dividend_raise", "dividend", "dividend_change_pct > 10", None, "structured"),
        ("dividend_special", "dividend", 'dividend_kind == "special"', None, "structured"),
        ("buyback_large", "buyback", large, None, "structured"),
        ("buyback_new_program", "buyback", "is_new_program == True", None, "structured"),
        ("dividend_framing_growth", "dividend", None,
         TextFeatureFilter(feature="dividend_framing", equals=framing), "text"),
        ("buyback_rationale_undervalued", "buyback", None,
         TextFeatureFilter(feature="buyback_rationale", equals=rationale), "text"),
    ]


def candidate_specs(
    events: pd.DataFrame, holding_days: int = DEFAULT_HOLDING_DAYS, seed: int = 0
) -> list[PlantSpec]:
    """Strengths x single/conjunction x structured/text, before the min_events filter.

    Conjunctions add small caps. Directions alternate across strengths (GRID_DIRECTIONS)
    so that matching a finding to a plant has to get the sign right.
    """
    specs = []
    for name, event_type, condition, text_feature, feature_type in _base_specs(events):
        for complexity in ("single", "conjunction"):
            cond = condition
            if complexity == "conjunction":
                small = 'mcap_bucket == "small"'
                cond = f"{condition} and {small}" if condition else small
            suffix = "_small" if complexity == "conjunction" else ""
            for effect, strength in STRENGTHS.items():
                direction = GRID_DIRECTIONS[effect]
                sign = "up" if direction > 0 else "down"
                specs.append(
                    PlantSpec(
                        spec_id=f"{name}{suffix}_{sign}{round(effect * 10_000)}bp",
                        event_type=event_type,
                        condition=cond,
                        text_feature=text_feature,
                        effect=effect,
                        holding_days=holding_days,
                        direction=direction,
                        seed=seed,
                        strength=strength,
                        complexity=complexity,
                        feature_type=feature_type,
                    )
                )
    return specs


def default_spec_grid(
    events: pd.DataFrame,
    min_events: int | None = None,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    seed: int = 0,
) -> tuple[list[PlantSpec], list[DroppedSpec]]:
    """(specs with at least min_events eligible targets in events, the dropped ones).

    min_events defaults to plant_min_events in the config.
    """
    if min_events is None:
        min_events = default_plant_min_events()
    specs, dropped = [], []
    for spec in candidate_specs(events, holding_days, seed):
        n_targets = len(select_targets(events, spec))
        if n_targets >= min_events:
            specs.append(spec)
        else:
            dropped.append(DroppedSpec(spec_id=spec.spec_id, n_targets=n_targets))
    return specs, dropped
