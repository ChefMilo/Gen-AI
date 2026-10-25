"""Planted universes: prices with a known abnormal return added after chosen events.

A PlantSpec picks target events (eligible events matching a structured condition
and/or a text feature) and an effect. make_planted adds direction * effect /
holding_days to `ret` on each of the holding_days trading days starting at the
event's first_tradeable_date, then rebuilds adj_close from the modified ret. Under
the reference event study (evaluation.reference_car), each target's CAR rises by
direction * effect, and nothing else moves. The truth records which events were
planted, for matching findings against planted signals.

Only `ret` and `adj_close` change. close, OHLC, raw_close and mcap are left as they
were, so the events (whose mcap fields come from the prices) stay valid.
"""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from store.schema import EventType
from universes.fake_events import BUYBACK_RATIONALE_LABELS, DIVIDEND_FRAMING_LABELS

DEFAULT_HOLDING_DAYS = 5
STRENGTHS = {0.005: "very_weak", 0.01: "weak", 0.02: "moderate", 0.04: "strong"}
GRID_DIRECTIONS = {0.005: 1, 0.01: -1, 0.02: 1, 0.04: -1}

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

Strength = Literal["very_weak", "weak", "moderate", "strong"]
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
    prices: pd.DataFrame, events: pd.DataFrame, spec: PlantSpec, min_events: int
) -> tuple[pd.DataFrame, PlantTruth]:
    """(prices with the effect planted, truth). Raises if fewer than min_events targets."""
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
    truth = PlantTruth(
        spec=spec,
        target_event_ids=list(targets["event_id"]),
        n_targets=len(targets),
        overlaps=overlaps,
        unmeasurable_event_ids=unmeasurable,
    )
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
    min_events: int,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    seed: int = 0,
) -> tuple[list[PlantSpec], list[DroppedSpec]]:
    """(specs with at least min_events eligible targets in events, the dropped ones)."""
    specs, dropped = [], []
    for spec in candidate_specs(events, holding_days, seed):
        n_targets = len(select_targets(events, spec))
        if n_targets >= min_events:
            specs.append(spec)
        else:
            dropped.append(DroppedSpec(spec_id=spec.spec_id, n_targets=n_targets))
    return specs, dropped
