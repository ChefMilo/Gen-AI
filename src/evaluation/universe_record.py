"""What Evaluation knows about a universe: its type, seed and planted truths.

A planted universe may carry several PlantSpecs, planted one after the other on the
same prices (plant_universe). Targets of different specs must not share a window
day on the same ticker (this includes one event targeted by two specs), or those
targets would carry both effects. plant_universe drops every target in such a clash
from its spec before planting (on_clash="drop", recorded in dropped_targets) or
raises (on_clash="raise"). Overlaps within one spec are allowed and are listed in
that truth's overlaps, as make_planted does.

After planting, each truth's injected_car is recomputed from the final prices minus
the original ones (recompute_injected), so it holds every effect that reached a
target's window, from any spec. spillover_event_ids is recomputed on all the events
(make_planted only saw those left after the drops, so a dropped target is a
spillover candidate too), and, given benchmarks, the baseline fields are measured on
the original prices.
"""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from universes.placebo import PlaceboInfo
from universes.planted import (
    PlantSpec,
    PlantTruth,
    baseline_fields,
    default_plant_min_events,
    injected_cars,
    make_planted,
    select_targets,
    spillover_ids,
    window_positions,
)

UniverseType = Literal["planted", "placebo", "real"]
OnClash = Literal["drop", "raise"]


class TargetRef(BaseModel):
    spec_id: str
    event_id: str


class DroppedTarget(BaseModel):
    """A target removed from its spec: its window shares days with another spec's target."""

    spec_id: str
    event_id: str
    ticker: str
    clashes_with: list[TargetRef]


class UniverseRecord(BaseModel):
    universe_id: str
    universe_type: UniverseType
    seed: int | None = None  # None for the real universe
    truths: list[PlantTruth] = []  # planted universes only
    dropped_targets: list[DroppedTarget] = []  # planted universes only
    placebo_info: PlaceboInfo | None = None  # placebo universes only

    @model_validator(mode="after")
    def _consistent(self) -> "UniverseRecord":
        if self.universe_type != "planted" and (self.truths or self.dropped_targets):
            raise ValueError(f"{self.universe_type} universe cannot have planted truths")
        if self.universe_type != "placebo" and self.placebo_info is not None:
            raise ValueError("placebo_info is only for placebo universes")
        spec_ids = [t.spec.spec_id for t in self.truths]
        if len(spec_ids) != len(set(spec_ids)):
            raise ValueError(f"duplicate spec_id in truths: {spec_ids}")
        return self


def measurable_target_ids(truth: PlantTruth) -> set[str]:
    """Targets the reference event study can measure: targets minus unmeasurable."""
    return set(truth.target_event_ids) - set(truth.unmeasurable_event_ids)


def planted_record(
    universe_id: str,
    truths: list[PlantTruth],
    seed: int | None = None,
    dropped_targets: list[DroppedTarget] | None = None,
) -> UniverseRecord:
    """Record for a planted universe; seed defaults to the specs' seed, which must agree."""
    if seed is None:
        seeds = {t.spec.seed for t in truths}
        if len(seeds) != 1:
            raise ValueError(f"specs have seeds {sorted(seeds)}; pass seed explicitly")
        seed = seeds.pop()
    return UniverseRecord(
        universe_id=universe_id, universe_type="planted", seed=seed, truths=truths,
        dropped_targets=dropped_targets or [],
    )


def placebo_record(universe_id: str, info: PlaceboInfo) -> UniverseRecord:
    """Record for a placebo universe from make_placebo's info."""
    return UniverseRecord(
        universe_id=universe_id, universe_type="placebo", seed=info.seed, placebo_info=info
    )


def real_record(universe_id: str) -> UniverseRecord:
    return UniverseRecord(universe_id=universe_id, universe_type="real")


def _windows(prices: pd.DataFrame, targets: pd.DataFrame, holding_days: int) -> dict:
    """event_id -> (ticker, first day, last day) of the rows make_planted would change.

    None when the ticker has no prices row on the first tradeable date (nothing planted).
    """
    dates = pd.to_datetime(prices["date"]).dt.normalize().to_numpy()
    by_ticker = {
        ticker: np.sort(dates[rows]) for ticker, rows in prices.groupby("ticker").indices.items()
    }
    day0s = pd.to_datetime(targets["first_tradeable_date"]).dt.normalize().to_numpy()
    out = {}
    for event_id, ticker, day0 in zip(targets["event_id"], targets["ticker"], day0s,
                                      strict=True):
        days = by_ticker.get(ticker, np.array([], dtype="datetime64[ns]"))
        pos = int(np.searchsorted(days, day0))
        if pos >= len(days) or days[pos] != day0:
            out[event_id] = None
            continue
        window = days[pos : pos + holding_days]
        out[event_id] = (ticker, window[0], window[-1])
    return out


def find_clashes(
    prices: pd.DataFrame, events: pd.DataFrame, specs: list[PlantSpec]
) -> list[DroppedTarget]:
    """Targets whose window shares a day with another spec's target on the same ticker.

    The same event targeted by two specs always clashes.
    """
    targets = [select_targets(events, spec) for spec in specs]
    windows = [_windows(prices, t, s.holding_days) for t, s in zip(targets, specs, strict=True)]
    tickers = [dict(zip(t["event_id"], t["ticker"], strict=True)) for t in targets]
    clashes: dict[tuple[int, str], list[TargetRef]] = {}
    for a in range(len(specs)):
        for b in range(a + 1, len(specs)):
            for ea, wa in windows[a].items():
                for eb, wb in windows[b].items():
                    same = ea == eb
                    overlap = (
                        wa is not None and wb is not None and wa[0] == wb[0]
                        and wa[1] <= wb[2] and wb[1] <= wa[2]
                    )
                    if same or overlap:
                        clashes.setdefault((a, ea), []).append(
                            TargetRef(spec_id=specs[b].spec_id, event_id=eb))
                        clashes.setdefault((b, eb), []).append(
                            TargetRef(spec_id=specs[a].spec_id, event_id=ea))
    return [
        DroppedTarget(spec_id=specs[k].spec_id, event_id=event_id,
                      ticker=tickers[k][event_id], clashes_with=refs)
        for (k, event_id), refs in sorted(clashes.items())
    ]


def recompute_injected(
    original: pd.DataFrame, final: pd.DataFrame, events: pd.DataFrame,
    truths: list[PlantTruth],
) -> list[PlantTruth]:
    """truths with injected_car from final["ret"] - original["ret"] over each target's window.

    original and final must have the same rows in the same order (make_planted keeps them).
    """
    if not original.index.equals(final.index):
        raise ValueError("original and final prices must have the same rows in the same order")
    original_ret = original["ret"].to_numpy(dtype=float)
    final_ret = final["ret"].to_numpy(dtype=float)
    out = []
    for truth in truths:
        targets = events[events["event_id"].isin(truth.target_event_ids)]
        windows = window_positions(original, targets, truth.spec.holding_days)
        injected, mean = injected_cars(original_ret, final_ret, windows,
                                       measurable_target_ids(truth))
        out.append(truth.model_copy(update={"injected_car": injected,
                                            "injected_car_mean": mean}))
    return out


def plant_universe(
    universe_id: str,
    prices: pd.DataFrame,
    events: pd.DataFrame,
    specs: list[PlantSpec],
    min_events: int | None = None,
    seed: int | None = None,
    on_clash: OnClash = "drop",
    benchmarks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, UniverseRecord]:
    """(prices with every spec planted in turn, record).

    Clashing targets across specs are dropped (recorded on the record) or raise, per
    on_clash. min_events (default: plant_min_events in the config) applies after the
    drops; make_planted raises below it. With benchmarks, each truth gets its
    baseline fields and snr from the unplanted prices.
    """
    if min_events is None:
        min_events = default_plant_min_events()
    dropped = find_clashes(prices, events, specs)
    if dropped and on_clash == "raise":
        listed = ", ".join(f"{d.spec_id}:{d.event_id}" for d in dropped)
        raise ValueError(f"targets of different specs share window days: {listed}")
    truths = []
    planted = prices
    for spec in specs:
        drop_ids = {d.event_id for d in dropped if d.spec_id == spec.spec_id}
        spec_events = events[~events["event_id"].isin(drop_ids)]
        planted, truth = make_planted(planted, spec_events, spec, min_events)
        truths.append(truth)
    truths = recompute_injected(prices, planted, events, truths)
    truths = [
        t.model_copy(update={
            "spillover_event_ids": spillover_ids(prices, events, t.target_event_ids,
                                                 t.spec.holding_days),
            **(baseline_fields(prices, benchmarks, events, t) if benchmarks is not None
               else {}),
        })
        for t in truths
    ]
    return planted, planted_record(universe_id, truths, seed, dropped)
