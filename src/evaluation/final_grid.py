"""The final planted grid, built from config, and the dev/final overlap guard.

The grid is code plus config (the grid: block of configs/final.yaml), not a hand list:

    strengths x complexities x feature types

Each (complexity, feature type) cell has ONE base, planted at every strength, so the
strength comparison is within a base and never confounded with it. Variants of a base
select the same targets, and evaluation.plan never puts two of them in one universe.
A base is a dsl.signal.SignalSpec without holding days or direction: event_type plus
conditions and/or a text filter. A single base uses one feature (one condition, or the
text filter); a conjunction base at least two. grid_specs checks that no part of a base
finds it on its own: the targets of the event type alone (single) or of each feature
alone (conjunction) must have Jaccard < jaccard_min with the base's targets, or a
system could match the plant without the feature the cell is meant to test.

build_grid returns, per (base, strength), the SignalSpec a system would have to define
to find the plant exactly (used by scripts/oracle_check.py) and the PlantSpec that
plants it; the PlantSpec's pandas condition is rendered from the same conditions, so
the two select the same events.

Overlap guard (check_dev_overlap, called by evaluation.plan.plan_experiment when the
config has a dev_guard: block), against every dev config:
- every universe seed of the final plan must be at least dev_guard.min_seed and outside
  the seeds the dev configs use;
- no final spec may have the canonical signature (spec_signature, not the name) of a
  dev spec.
And against the specs of agent dev configs (agent.experiment's {experiment, dev} format,
the specs the agent systems were developed on), measured on the final plan's events:
- containment: no final base may share more than dev_guard.max_containment of its
  targets with any one dev spec (shared / final targets). This catches a final base
  nested in a broader dev spec, which Jaccard misses (a base of 80 targets inside a dev
  spec of 220 has Jaccard 0.36);
- Jaccard: nor reach jaccard_min with one, the overlap at which a finding of the dev spec
  would match the final plant. Jaccard <= containment, so this only binds when
  max_containment >= jaccard_min.
The runner-test grids (evaluation.plan.ExperimentConfig files such as fake-v1, run with
the LLM-free systems only) get the seed and signature checks, not the overlap ones:
the final bases are cut from the same fake events, so they nest in those grids by
construction.
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
    eligible,
    select_targets,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Strength label per effect size (universes.planted.STRENGTHS plus 4%).
STRENGTH_LABELS = {0.0025: "very_weak", 0.005: "weak", 0.01: "moderate", 0.02: "strong",
                   0.04: "very_strong"}
COMPLEXITIES = ("single", "conjunction")
FEATURE_TYPES = ("structured", "text")


class GridBase(BaseModel):
    """One base hypothesis: a SignalSpec's event selection, with a name."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    event_type: Literal["dividend", "buyback"]
    conditions: list[Condition] = []
    text_filter: TextFilter | None = None

    @property
    def n_features(self) -> int:
        return len(self.conditions) + int(self.text_filter is not None)


class GridConfig(BaseModel):
    """The grid: block of a final experiment config."""

    model_config = ConfigDict(extra="forbid")

    strengths: list[float] = Field(min_length=1)
    directions: list[Literal[1, -1]]  # one per strength
    # complexity -> feature type -> the cell's base
    bases: dict[Literal["single", "conjunction"], dict[Literal["structured", "text"], GridBase]]

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
        names = [b.name for cells in self.bases.values() for b in cells.values()]
        if len(names) != len(set(names)):
            raise ValueError("base names must be unique")
        for complexity, cells in self.bases.items():
            for feature_type, base in cells.items():
                where = f"{complexity}/{feature_type} base {base.name}"
                if (feature_type == "text") != (base.text_filter is not None):
                    raise ValueError(f"{where}: text bases (only) need a text_filter")
                if complexity == "single" and base.n_features != 1:
                    raise ValueError(f"{where}: a single base uses exactly one feature")
                if complexity == "conjunction" and base.n_features < 2:
                    raise ValueError(f"{where}: a conjunction base needs two features or more")
        return self

    def cells(self) -> list[tuple[str, str, GridBase]]:
        """(complexity, feature type, base) in a fixed order."""
        return [(c, f, self.bases[c][f]) for f in FEATURE_TYPES for c in COMPLEXITIES
                if f in self.bases.get(c, {})]


class GridCell(BaseModel):
    """One spec of the grid: the plant and the signal that recovers it exactly."""

    plant: PlantSpec
    signal: SignalSpec


class DevGuardConfig(BaseModel):
    """The dev_guard: block: what the final plan must not share with development."""

    model_config = ConfigDict(extra="forbid")

    min_seed: int  # every final seed is >= this; dev seeds stay below it
    # Relative to the repo root, written with /; missing files are skipped. Strings, not
    # Paths, so plan.json (and its hash) is the same on every OS.
    dev_configs: list[str] = []
    # Most of a final base's targets one agent dev spec may share (module docstring).
    max_containment: float = Field(0.05, ge=0, le=1)


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


def _plant(base: GridBase, conditions, text, effect=0.01, direction=1, holding_days=5,
           seed=0, complexity="single", feature_type="structured",
           spec_id=None) -> PlantSpec:
    return PlantSpec(
        spec_id=spec_id or base.name, event_type=base.event_type,
        condition=render_condition(conditions),
        text_feature=(TextFeatureFilter(feature=text.feature, equals=text.equals)
                      if text else None),
        effect=effect, holding_days=holding_days, direction=direction, seed=seed,
        strength=STRENGTH_LABELS.get(effect, "moderate"), complexity=complexity,
        feature_type=feature_type,
    )


def build_grid(grid: GridConfig, holding_days: int, seed: int = 0) -> list[GridCell]:
    """Every (base, strength), in a fixed order: feature type, complexity, strength."""
    out = []
    for complexity, feature_type, base in grid.cells():
        for effect, direction in zip(grid.strengths, grid.directions, strict=True):
            sign = "up" if direction > 0 else "down"
            plant = _plant(base, base.conditions, base.text_filter, effect, direction,
                           holding_days, seed, complexity, feature_type,
                           spec_id=f"{base.name}_{sign}{_bp(effect)}bp")
            signal = SignalSpec(
                event_type=base.event_type, conditions=list(base.conditions),
                text_filter=base.text_filter, holding_days=holding_days,
                direction="long" if direction > 0 else "short",
            )
            out.append(GridCell(plant=plant, signal=signal))
    return out


def _parts(base: GridBase) -> list[tuple[str, list, TextFilter | None]]:
    """What a system could use instead of the whole base: the event type alone for a
    single base; each feature alone for a conjunction."""
    if base.n_features == 1:
        return [("event type alone", [], None)]
    parts = [(f"{c.field} alone", [c], None) for c in base.conditions]
    if base.text_filter is not None:
        parts.append((f"{base.text_filter.feature} alone", [], base.text_filter))
    return parts


def _ids(events: pd.DataFrame, base: GridBase, conditions, text) -> set:
    if not conditions and text is None:
        rows = eligible(events)
        return set(rows.loc[rows["event_type"] == base.event_type, "event_id"])
    return set(select_targets(events, _plant(base, conditions, text))["event_id"])


def base_report(grid: GridConfig, events: pd.DataFrame) -> pd.DataFrame:
    """Per cell: eligible targets, small-cap share, and the largest Jaccard any part of
    the base reaches alone (see the module docstring)."""
    rows = []
    for complexity, feature_type, base in grid.cells():
        targets = select_targets(events, _plant(base, base.conditions, base.text_filter))
        ids = set(targets["event_id"])
        worst, worst_part = 0.0, ""
        for name, conditions, text in _parts(base):
            other = _ids(events, base, conditions, text)
            union = ids | other
            jaccard = len(ids & other) / len(union) if union else 0.0
            if jaccard >= worst:
                worst, worst_part = jaccard, name
        small = float((targets["mcap_bucket"] == "small").mean()) if len(ids) else 0.0
        rows.append({
            "complexity": complexity, "feature_type": feature_type, "base": base.name,
            "n_targets": len(ids), "small_share": small,
            "max_part_jaccard": worst, "part": worst_part,
        })
    return pd.DataFrame(rows)


def grid_specs(grid: GridConfig, events: pd.DataFrame, min_events: int, holding_days: int,
               seed: int = 0, jaccard_min: float | None = None) -> list[PlantSpec]:
    """The grid's PlantSpecs. Raises if a text label is not in the data's label set, a
    base has fewer than min_events eligible targets, or (given jaccard_min) a part of a
    base matches it alone: the final grid is never trimmed."""
    for _, _, base in grid.cells():
        if base.text_filter is not None:
            labels = FEATURE_LABELS[base.text_filter.feature]
            if base.text_filter.equals not in labels:
                raise ValueError(f"base {base.name}: label {base.text_filter.equals!r} "
                                 f"is not one of {labels}")
    report = base_report(grid, events)
    short = report[report["n_targets"] < min_events]
    if len(short):
        raise ValueError(f"grid bases with fewer than plant_min_events={min_events} eligible "
                         f"targets: {dict(zip(short['base'], short['n_targets'], strict=True))}")
    if jaccard_min is not None:
        leaky = report[report["max_part_jaccard"] >= jaccard_min]
        if len(leaky):
            raise ValueError("grid bases a part of which matches them alone (Jaccard >= "
                             f"jaccard_min={jaccard_min}): "
                             + ", ".join(f"{r.base} ({r.part}: {r.max_part_jaccard:.2f})"
                                         for r in leaky.itertuples()))
    return [cell.plant for cell in build_grid(grid, holding_days, seed)]


# --- Dev overlap guard ------------------------------------------------------------------


class DevUsage(BaseModel):
    seeds: set[int] = set()
    specs: dict[tuple, str] = {}  # signature -> "<config>:<spec_id>"
    # Agent dev specs, checked for overlap: ("<config>:<spec_id>", spec)
    agent_specs: list[tuple[str, PlantSpec]] = []


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
            usage.agent_specs += [(f"{path}:{s.spec_id}", s) for s in specs]
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


OVERLAP_COLUMNS = ["final_spec", "n_variants", "dev_spec", "n_final", "n_dev", "n_shared",
                   "containment", "jaccard"]


def _target_ids(events: pd.DataFrame, spec: PlantSpec) -> set:
    return set(select_targets(events, spec)["event_id"])


def dev_overlap(specs: list[PlantSpec], dev_specs: list[tuple[str, PlantSpec]],
                events: pd.DataFrame) -> pd.DataFrame:
    """One row per (final base, agent dev spec): targets on these events, shared targets,
    containment (shared / final targets) and Jaccard. A base's variants (strengths,
    directions) select the same targets, so each base is one row, named by its first
    spec."""
    bases: dict[tuple, list[PlantSpec]] = {}
    for spec in specs:
        text = (spec.text_feature.feature, spec.text_feature.equals) if spec.text_feature else None
        bases.setdefault((spec.event_type, _normal_conjuncts(spec.condition), text),
                         []).append(spec)
    dev_ids = [(name, _target_ids(events, s)) for name, s in dev_specs]
    rows = []
    for variants in bases.values():
        ids = _target_ids(events, variants[0])
        for name, other in dev_ids:
            shared = len(ids & other)
            union = len(ids | other)
            rows.append({
                "final_spec": variants[0].spec_id, "n_variants": len(variants),
                "dev_spec": name, "n_final": len(ids), "n_dev": len(other),
                "n_shared": shared,
                "containment": shared / len(ids) if ids else 0.0,
                "jaccard": shared / union if union else 0.0,
            })
    return pd.DataFrame(rows, columns=OVERLAP_COLUMNS)


def check_dev_overlap(seeds: set[int], specs: list[PlantSpec], guard: DevGuardConfig,
                      events: pd.DataFrame, jaccard_min: float | None = None) -> None:
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
    overlap = dev_overlap(specs, usage.agent_specs, events)
    bad = overlap["containment"] > guard.max_containment
    if jaccard_min is not None:
        bad |= overlap["jaccard"] >= jaccard_min
    if bad.any():
        raise ValueError(
            "final bases that overlap agent dev specs (containment > dev_guard."
            f"max_containment={guard.max_containment}"
            + ("" if jaccard_min is None else f" or Jaccard >= jaccard_min={jaccard_min}")
            + "): " + "; ".join(
                f"{r.final_spec} vs {r.dev_spec}: {r.n_shared}/{r.n_final} shared "
                f"(containment {r.containment:.2f}, Jaccard {r.jaccard:.2f})"
                for r in overlap[bad].itertuples()))
