"""The final planted grid, built from config, and the dev/final overlap guard.

The grid is code plus config (the grid: block of configs/final.yaml), not a hand list:

    strengths x complexities x feature types

Each feature type lists its bases, one per strength in order, so every cell gets one
spec and a base is never planted at two strengths (which would give nested
near-duplicates). A base is a dsl.signal.SignalSpec without holding days or direction:
event_type plus conditions and/or a text filter. The conjunction level adds the
grid's conjunction conditions (small caps) to the base. build_grid returns, per cell,
the SignalSpec a system would have to define to find the plant exactly (used by
scripts/oracle_check.py) and the PlantSpec that plants it; the PlantSpec's pandas
condition is rendered from the same conditions, so the two select the same events.

Overlap guard (check_dev_overlap, called by evaluation.plan.plan_experiment when the
config has a dev_guard: block): every universe seed of the final plan must be at least
dev_guard.min_seed and outside the seeds the dev configs use, and no final spec may
have the canonical signature (spec_signature, not the name) of a dev spec.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsl.condition import Condition
from dsl.signal import SignalSpec, TextFilter
from universes.planted import (
    FEATURE_LABELS,
    PlantSpec,
    TextFeatureFilter,
    select_targets,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Strength label per effect size (universes.planted.STRENGTHS plus 4%).
STRENGTH_LABELS = {0.0025: "very_weak", 0.005: "weak", 0.01: "moderate", 0.02: "strong",
                   0.04: "very_strong"}


class GridBase(BaseModel):
    """One base hypothesis: a SignalSpec's event selection, with a name."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    event_type: Literal["dividend", "buyback"]
    conditions: list[Condition] = []
    text_filter: TextFilter | None = None

    @model_validator(mode="after")
    def _selects_something(self) -> "GridBase":
        if not self.conditions and self.text_filter is None:
            raise ValueError(f"base {self.name}: needs conditions or a text_filter")
        return self


class GridConfig(BaseModel):
    """The grid: block of a final experiment config."""

    model_config = ConfigDict(extra="forbid")

    strengths: list[float] = Field(min_length=1)
    directions: list[Literal[1, -1]]  # one per strength
    complexities: list[Literal["single", "conjunction"]] = ["single", "conjunction"]
    conjunction: list[Condition] = Field(min_length=1)  # added at the conjunction level
    bases: dict[Literal["structured", "text"], list[GridBase]]

    @model_validator(mode="after")
    def _shape(self) -> "GridConfig":
        n = len(self.strengths)
        if len(set(self.strengths)) != n:
            raise ValueError("strengths must be distinct")
        unknown = [s for s in self.strengths if s not in STRENGTH_LABELS]
        if unknown:
            raise ValueError(f"no strength label for {unknown}; known {sorted(STRENGTH_LABELS)}")
        if len(self.directions) != n:
            raise ValueError("one direction per strength")
        names = [b.name for bases in self.bases.values() for b in bases]
        if len(names) != len(set(names)):
            raise ValueError("base names must be unique")
        for feature_type, bases in self.bases.items():
            if len(bases) != n:
                raise ValueError(f"{feature_type}: one base per strength ({n}), got {len(bases)}")
            for base in bases:
                if (feature_type == "text") != (base.text_filter is not None):
                    raise ValueError(f"base {base.name}: text bases (only) need a text_filter")
                if feature_type == "text" and base.conditions:
                    raise ValueError(f"text base {base.name}: no structured conditions")
        return self


class GridCell(BaseModel):
    """One cell of the grid: the plant and the signal that recovers it exactly."""

    plant: PlantSpec
    signal: SignalSpec


class DevGuardConfig(BaseModel):
    """The dev_guard: block: what the final plan must not share with development."""

    model_config = ConfigDict(extra="forbid")

    min_seed: int  # every final seed is >= this; dev seeds stay below it
    # Relative to the repo root, written with /; missing files are skipped. Strings, not
    # Paths, so plan.json (and its hash) is the same on every OS.
    dev_configs: list[str] = []


# --- Rendering and signatures ----------------------------------------------------------


def _value(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return f'"{value}"'
    return f"{value:g}"


def render_condition(conditions: list[Condition]) -> str | None:
    """The pandas query universes.planted.select_targets runs for these conditions."""
    if not conditions:
        return None
    return " and ".join(f"{c.field} {c.op} {_value(c.value)}" for c in conditions)


def _normal_conjuncts(condition: str | None) -> tuple[str, ...]:
    if not condition:
        return ()
    parts = [" ".join(p.split()) for p in condition.replace("'", '"').split(" and ")]
    return tuple(sorted(p for p in parts if p))


def spec_signature(spec: PlantSpec) -> tuple:
    """What a plant is, whatever it is called: event type, conditions (order and
    whitespace ignored), text filter, effect, direction and holding days."""
    text = (spec.text_feature.feature, spec.text_feature.equals) if spec.text_feature else None
    return (spec.event_type, _normal_conjuncts(spec.condition), text,
            round(spec.effect, 10), spec.direction, spec.holding_days)


# --- The grid ---------------------------------------------------------------------------


def _bp(effect: float) -> int:
    return round(effect * 10_000)


def build_grid(grid: GridConfig, holding_days: int, seed: int = 0) -> list[GridCell]:
    """Every cell, in a fixed order: feature type, complexity, strength."""
    cells = []
    for feature_type in ("structured", "text"):
        for complexity in grid.complexities:
            for k, (effect, direction) in enumerate(zip(grid.strengths, grid.directions,
                                                        strict=True)):
                base = grid.bases[feature_type][k]
                conditions = list(base.conditions)
                if complexity == "conjunction":
                    conditions += list(grid.conjunction)
                sign = "up" if direction > 0 else "down"
                suffix = "_small" if complexity == "conjunction" else ""
                spec_id = f"{base.name}{suffix}_{sign}{_bp(effect)}bp"
                text = base.text_filter
                plant = PlantSpec(
                    spec_id=spec_id, event_type=base.event_type,
                    condition=render_condition(conditions),
                    text_feature=(TextFeatureFilter(feature=text.feature, equals=text.equals)
                                  if text else None),
                    effect=effect, holding_days=holding_days, direction=direction, seed=seed,
                    strength=STRENGTH_LABELS[effect], complexity=complexity,
                    feature_type=feature_type,
                )
                signal = SignalSpec(
                    event_type=base.event_type, conditions=conditions, text_filter=text,
                    holding_days=holding_days, direction="long" if direction > 0 else "short",
                )
                cells.append(GridCell(plant=plant, signal=signal))
    return cells


def grid_specs(grid: GridConfig, events: pd.DataFrame, min_events: int, holding_days: int,
               seed: int = 0) -> list[PlantSpec]:
    """The grid's PlantSpecs. Raises if a text label is not in the data's label set or a
    cell has fewer than min_events eligible targets: the final grid is never trimmed."""
    for bases in grid.bases.values():
        for base in bases:
            if base.text_filter is not None:
                labels = FEATURE_LABELS[base.text_filter.feature]
                if base.text_filter.equals not in labels:
                    raise ValueError(f"base {base.name}: label {base.text_filter.equals!r} "
                                     f"is not one of {labels}")
    specs, short = [], []
    for cell in build_grid(grid, holding_days, seed):
        n = len(select_targets(events, cell.plant))
        if n < min_events:
            short.append(f"{cell.plant.spec_id} ({n})")
        specs.append(cell.plant)
    if short:
        raise ValueError(f"grid cells with fewer than plant_min_events={min_events} "
                         f"eligible targets: {', '.join(short)}")
    return specs


# --- Dev overlap guard ------------------------------------------------------------------


class DevUsage(BaseModel):
    seeds: set[int] = set()
    specs: dict[tuple, str] = {}  # signature -> "<config>:<spec_id>"


def _seeds_of(base_seed: int, n_planted: int, n_placebo: int) -> set[int]:
    from evaluation.plan import PLACEBO_SEED_OFFSET

    planted = range(base_seed, base_seed + n_planted)
    offset = base_seed + PLACEBO_SEED_OFFSET
    return {base_seed, *planted, *range(offset, offset + n_placebo)}


def dev_usage(paths: list[str], events: pd.DataFrame) -> DevUsage:
    """Seeds and spec signatures the dev configs use.

    Two formats: an evaluation.plan.ExperimentConfig (its planned seeds, and its grid:
    or else universes.planted.default_spec_grid's candidates on these events), or
    agent.experiment's {experiment, dev} file (its base seed, its dev universe seeds
    and specs).
    """
    from evaluation.plan import ExperimentConfig
    from universes.planted import candidate_specs

    usage = DevUsage()
    for path in paths:
        full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
        if not full.exists():
            log.warning("dev_guard: %s not found; its seeds and specs are not checked", path)
            continue
        data = yaml.safe_load(full.read_text(encoding="utf-8"))
        specs: list[PlantSpec] = []
        if "dev" in data:
            exp = ExperimentConfig.model_validate(data["experiment"])
            usage.seeds |= _seeds_of(exp.base_seed, exp.n_planted, exp.n_placebo)
            for kind in ("planted", "placebo"):
                for universe in data["dev"].get(kind, []):
                    usage.seeds.add(int(universe["seed"]))
                    specs += [PlantSpec.model_validate({"holding_days": exp.holding_days, **s})
                              for s in universe.get("specs", [])]
        else:
            exp = ExperimentConfig.model_validate(data)
            usage.seeds |= _seeds_of(exp.base_seed, exp.n_planted, exp.n_placebo)
            if exp.grid is not None:
                specs = [c.plant for c in build_grid(exp.grid, exp.holding_days)]
            else:
                specs = candidate_specs(events, exp.holding_days)
        for spec in specs:
            usage.specs.setdefault(spec_signature(spec), f"{path}:{spec.spec_id}")
    return usage


def check_dev_overlap(seeds: set[int], specs: list[PlantSpec], guard: DevGuardConfig,
                      events: pd.DataFrame) -> None:
    """Raise if a final seed or spec is a dev one (see the module docstring)."""
    low = sorted(s for s in seeds if s < guard.min_seed)
    if low:
        raise ValueError(f"final seeds below dev_guard.min_seed={guard.min_seed}: {low[:10]}")
    usage = dev_usage(guard.dev_configs, events)
    above = sorted(s for s in usage.seeds if s >= guard.min_seed)
    if above:
        raise ValueError(f"dev configs use seeds >= dev_guard.min_seed={guard.min_seed}: "
                         f"{above[:10]}; dev seeds must stay below it")
    shared = sorted(seeds & usage.seeds)
    if shared:
        raise ValueError(f"final seeds also used by dev configs: {shared[:10]}")
    clashes = [f"{s.spec_id} = {usage.specs[spec_signature(s)]}" for s in specs
               if spec_signature(s) in usage.specs]
    if clashes:
        raise ValueError("final specs identical to dev specs (same signature): "
                         + "; ".join(clashes))

