"""What Evaluation knows about a universe: its type, seed and planted truths.

A planted universe may carry several PlantSpecs. They are planted one after the other
on the same prices (plant_universe), so a target of two specs, or a target whose
window overlaps another spec's target, gets both effects; each truth's overlaps only
list overlaps within that spec.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, model_validator

from universes.placebo import PlaceboInfo
from universes.planted import PlantSpec, PlantTruth, make_planted

UniverseType = Literal["planted", "placebo", "real"]


class UniverseRecord(BaseModel):
    universe_id: str
    universe_type: UniverseType
    seed: int | None = None  # None for the real universe
    truths: list[PlantTruth] = []  # planted universes only
    placebo_info: PlaceboInfo | None = None  # placebo universes only

    @model_validator(mode="after")
    def _consistent(self) -> "UniverseRecord":
        if self.universe_type != "planted" and self.truths:
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
    universe_id: str, truths: list[PlantTruth], seed: int | None = None
) -> UniverseRecord:
    """Record for a planted universe; seed defaults to the specs' seed, which must agree."""
    if seed is None:
        seeds = {t.spec.seed for t in truths}
        if len(seeds) != 1:
            raise ValueError(f"specs have seeds {sorted(seeds)}; pass seed explicitly")
        seed = seeds.pop()
    return UniverseRecord(
        universe_id=universe_id, universe_type="planted", seed=seed, truths=truths
    )


def placebo_record(universe_id: str, info: PlaceboInfo) -> UniverseRecord:
    """Record for a placebo universe from make_placebo's info."""
    return UniverseRecord(
        universe_id=universe_id, universe_type="placebo", seed=info.seed, placebo_info=info
    )


def real_record(universe_id: str) -> UniverseRecord:
    return UniverseRecord(universe_id=universe_id, universe_type="real")


def plant_universe(
    universe_id: str,
    prices: pd.DataFrame,
    events: pd.DataFrame,
    specs: list[PlantSpec],
    min_events: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, UniverseRecord]:
    """(prices with every spec planted in turn, record). Raises as make_planted does."""
    truths = []
    for spec in specs:
        prices, truth = make_planted(prices, events, spec, min_events)
        truths.append(truth)
    return prices, planted_record(universe_id, truths, seed)
