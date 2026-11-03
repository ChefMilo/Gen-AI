"""Experiment plan: which universes to build, from which specs and seeds.

plan_experiment(prices, events, cfg) is pure: the same data, config and base_seed
always give the same plan. It holds no frames, only Pydantic data (saved as
plan.json), and a fingerprint of the input data so that cached universes built from
other data are never reused (evaluation.universe_store).

Planted universes. default_spec_grid gives the kept specs. plant_universe drops every
target that clashes with another spec's target (evaluation.universe_record), and a
dropped target leaves the truth, so a finding on the spec's own definition scores
Jaccard = the share kept. Hard rules, per spec in each universe:
- at least plant_min_events targets left after the drops (plant_universe raises
  otherwise);
- at most max_dropped_share of its targets dropped (default 0.1).
Specs with nested target sets (a base spec at another strength, its small-cap
conjunction, or e.g. dividend raises that are all framed as growth) break these rules
together, so they never share a universe. specs_per_universe is therefore a maximum:
on data where most specs conflict, universes hold fewer.

Assignment (deterministic per base_seed, ClashIndex counts = find_clashes counts):
1. cover: every kept spec once, most-conflicted first, each into the emptiest
   universe it fits, then the one where it adds least cost; raises if a spec fits
   nowhere (then n_planted is too small for the data);
2. fill: free slots, always with a spec of the fewest appearances, so appearances
   differ by at most one;
3. improve: swap specs between universes while the summed cost drops. The cost, in
   lexicographic order: specs over the hard limits, repeated strengths, a missing
   feature type, clashing targets.

With a grid: block (configs/final.yaml) the kept specs are evaluation.final_grid's
cells instead of default_spec_grid, all of them (a cell short of targets raises), and a
dev_guard: block makes plan_experiment raise if a seed or spec is shared with dev.

Placebo universes use seeds base_seed + 1000 + i; planted universe i uses base_seed + i
(its specs carry that seed). There is one "real" universe, with no truths, for later
holdout work.

Universe ids are opaque (<experiment_id>-uNN, in a seeded random order over all
universes), because a research system sees its universe_id (evaluation.systems) and
must not learn the universe type from it. plan.json maps ids to types.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from evaluation.final_grid import DevGuardConfig, GridConfig, check_dev_overlap, grid_specs
from evaluation.universe_record import UniverseType
from universes.placebo import DEFAULT_MIN_GAP_DAYS
from universes.planted import (
    DEFAULT_HOLDING_DAYS,
    DroppedSpec,
    PlantSpec,
    default_spec_grid,
    eligible,
    select_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
EXPERIMENT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"

PLACEBO_SEED_OFFSET = 1000
_PLAN_STREAM = 7

# Positions per ticker in the (ticker, position) keys: more than any ticker's rows.
_ROW_SPAN = 1 << 32


class DataConfig(BaseModel):
    """Where the base prices and events come from.

    synthetic: universes.synthetic prices and benchmarks with fake events.
    real_prices_fake_events: store.prices (needs $GENAI_DATA_DIR) with FAKE events,
    until Data's event loader exists.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["synthetic", "real_prices_fake_events"] = "synthetic"
    n_tickers: int = Field(120, ge=1)  # synthetic only
    start: date = date(2015, 1, 1)  # synthetic only
    end: date = date(2026, 12, 31)  # synthetic only
    seed: int = 0  # synthetic prices and benchmarks
    events_seed: int = 0  # make_fake_events


class SystemsConfig(BaseModel):
    """The systems: block of an experiment config, in its mapping form.

    names: systems to run (overridden by --systems on the CLI);
    modules: imported before the names are looked up; each registers its systems with
    evaluation.systems.register_system.
    """

    model_config = ConfigDict(extra="forbid")

    names: list[str] = []
    modules: list[str] = []


class ExperimentConfig(BaseModel):
    """configs/experiment.yaml. Fields left None are filled by resolved()."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    n_planted: int = Field(10, ge=0)
    n_placebo: int = Field(10, ge=0)
    specs_per_universe: int = Field(4, ge=1)
    base_seed: int = 0
    holding_days: int = Field(DEFAULT_HOLDING_DAYS, ge=1)
    include_real: bool = True
    # A list of names, or a SystemsConfig mapping with names and modules.
    systems: list[str] | SystemsConfig = []
    n_boot: int = Field(2000, ge=1)
    # None: from configs/default.yaml (plant_min_events, backtest_budget, jaccard_min,
    # research_start, research_end) or universes.placebo (min_gap_days).
    plant_min_events: int | None = Field(None, ge=1)
    min_gap_days: int | None = Field(None, ge=1)
    backtest_budget: int | None = Field(None, ge=0)
    jaccard_min: float | None = Field(None, gt=0, le=1)
    research_start: date | None = None
    research_end: date | None = None
    # Most of a spec's targets that cross-spec clashes may drop. A dropped target leaves
    # the truth, so a finding on the spec's own definition scores Jaccard = kept share.
    max_dropped_share: float = Field(0.1, ge=0, le=1)
    data: DataConfig = DataConfig()
    # Final experiments: the planted grid built from config (evaluation.final_grid)
    # instead of default_spec_grid, the dev overlap guard, and the frozen plan hash.
    grid: GridConfig | None = None
    dev_guard: DevGuardConfig | None = None
    plan_sha256_file: str | None = None  # relative to the repo root

    @property
    def system_names(self) -> list[str]:
        s = self.systems
        return list(s.names if isinstance(s, SystemsConfig) else s)

    @property
    def system_modules(self) -> list[str]:
        return list(self.systems.modules) if isinstance(self.systems, SystemsConfig) else []

    def resolved(self, defaults_path: Path = DEFAULT_CONFIG_PATH) -> "ExperimentConfig":
        """A copy with every None default filled in (research_end may stay None)."""
        with open(defaults_path, encoding="utf-8") as f:
            defaults = yaml.safe_load(f)
        fill = {
            "plant_min_events": int(defaults["plant_min_events"]),
            "min_gap_days": DEFAULT_MIN_GAP_DAYS,
            "backtest_budget": int(defaults["backtest_budget"]),
            "jaccard_min": float(defaults["jaccard_min"]),
            "research_start": defaults.get("research_start"),
            "research_end": defaults.get("research_end"),
        }
        update = {k: v for k, v in fill.items() if getattr(self, k) is None}
        return self.model_validate({**self.model_dump(), **update})


def load_experiment_config(path: Path = EXPERIMENT_CONFIG_PATH) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))


class DataFingerprint(BaseModel):
    """Enough of the input data to tell when a cache was built from other data."""

    n_price_rows: int
    n_tickers: int
    price_start: date | None
    price_end: date | None
    prices_hash: str
    n_benchmark_rows: int
    benchmarks_hash: str
    n_events: int
    n_eligible: int
    event_start: date | None
    event_end: date | None
    event_ids_hash: str  # sorted event_ids
    events_hash: str  # every column, in row order


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def frame_hash(df: pd.DataFrame) -> str:
    """Hash of a frame's columns, values and row order (not its index)."""
    payload = df.reset_index(drop=True).to_json(
        orient="split", date_format="iso", date_unit="ns", default_handler=str
    )
    return _sha(payload.encode())


def prices_hash(prices: pd.DataFrame) -> str:
    """frame_hash for numeric-heavy frames: hashes values row by row, much faster."""
    rows = pd.util.hash_pandas_object(prices, index=False).to_numpy()
    return _sha(",".join(map(str, prices.columns)).encode() + rows.tobytes())


def _date_range(values) -> tuple[date | None, date | None]:
    days = pd.to_datetime(pd.Series(values)).dropna()
    if days.empty:
        return None, None
    return days.min().date(), days.max().date()


def data_fingerprint(
    prices: pd.DataFrame, benchmarks: pd.DataFrame, events: pd.DataFrame
) -> DataFingerprint:
    price_start, price_end = _date_range(prices["date"])
    event_start, event_end = _date_range(events["first_tradeable_date"])
    return DataFingerprint(
        n_price_rows=len(prices),
        n_tickers=int(prices["ticker"].nunique()),
        price_start=price_start,
        price_end=price_end,
        prices_hash=prices_hash(prices),
        n_benchmark_rows=len(benchmarks),
        benchmarks_hash=prices_hash(benchmarks),
        n_events=len(events),
        n_eligible=len(eligible(events)),
        event_start=event_start,
        event_end=event_end,
        event_ids_hash=_sha("\n".join(sorted(events["event_id"])).encode()),
        events_hash=frame_hash(events),
    )


class UniverseEntry(BaseModel):
    """One universe to build. Everything the build depends on, apart from the data."""

    model_config = ConfigDict(frozen=True)

    universe_id: str
    universe_type: UniverseType
    seed: int | None = None  # None for the real universe
    holding_days: int
    specs: list[PlantSpec] = []  # planted only, in planting order
    min_events: int | None = None  # planted only: plant_min_events
    min_gap_days: int | None = None  # placebo only

    def cache_key(self, fingerprint: DataFingerprint, version: int) -> str:
        """Changes when the entry, the data or the universe format changes."""
        payload = json.dumps(
            [version, self.model_dump(mode="json"), fingerprint.model_dump(mode="json")],
            sort_keys=True,
        )
        return _sha(payload.encode())


class ExperimentPlan(BaseModel):
    experiment_id: str
    base_seed: int
    config: ExperimentConfig  # resolved
    fingerprint: DataFingerprint
    kept_specs: list[PlantSpec]
    dropped_specs: list[DroppedSpec]
    appearances: dict[str, int]  # spec_id -> planted universes it is in
    planned_clashes: dict[str, int]  # planted universe_id -> targets find_clashes drops
    universes: list[UniverseEntry]

    def entry(self, universe_id: str) -> UniverseEntry:
        return next(u for u in self.universes if u.universe_id == universe_id)

    def by_type(self, universe_type: UniverseType) -> list[UniverseEntry]:
        return [u for u in self.universes if u.universe_type == universe_type]


# --- Clash counts ---------------------------------------------------------------


def _target_key(spec: PlantSpec) -> tuple:
    """Specs with equal keys select the same targets with the same windows."""
    text = spec.text_feature.model_dump_json() if spec.text_feature else None
    return (spec.event_type, spec.condition, text, spec.holding_days)


class ClashIndex:
    """Window clashes between target sets, with the definition of find_clashes.

    A target of spec a clashes with spec b if b targets the same event, or a target of
    b on the same ticker has a window sharing a trading day with a's window. Windows
    are the holding_days rows from first_tradeable_date on the ticker's prices rows,
    cut short at the end of the data; a target with no row on that day has no window
    and clashes only as the same event. dropped(...) gives the targets find_clashes
    would list, without its pairwise Python loops.
    """

    def __init__(self, prices: pd.DataFrame, events: pd.DataFrame, specs: list[PlantSpec]):
        tickers = pd.Index(sorted(pd.unique(prices["ticker"])))
        codes = tickers.get_indexer(prices["ticker"])
        days = pd.to_datetime(prices["date"]).dt.normalize().to_numpy(dtype="datetime64[D]")
        order = np.lexsort((days, codes))
        codes, days = codes[order], days.astype(np.int64)[order]
        group_start = np.searchsorted(codes, np.arange(len(tickers)), side="left")
        group_end = np.searchsorted(codes, np.arange(len(tickers)), side="right")
        day_keys = codes.astype(np.int64) * _ROW_SPAN + days

        self.keys = [_target_key(s) for s in specs]
        self._sets: dict[tuple, dict[str, np.ndarray]] = {}
        for spec, key in zip(specs, self.keys, strict=True):
            if key in self._sets:
                continue
            targets = select_targets(events, spec)
            t_codes = tickers.get_indexer(targets["ticker"])
            t_days = pd.to_datetime(targets["first_tradeable_date"]).dt.normalize()
            t_keys = t_codes.astype(np.int64) * _ROW_SPAN + t_days.to_numpy(
                dtype="datetime64[D]").astype(np.int64)
            pos = np.searchsorted(day_keys, t_keys)
            found = (t_codes >= 0) & (pos < len(day_keys))
            found[found] = day_keys[pos[found]] == t_keys[found]
            safe = np.where(t_codes >= 0, t_codes, 0)
            start = np.where(found, pos, -1)
            end = np.where(found, np.minimum(pos + spec.holding_days, group_end[safe]) - 1, -1)
            self._sets[key] = {
                "ids": targets["event_id"].to_numpy(dtype=object),
                "start": start, "end": end, "ticker_start": group_start[safe],
                "sorted_starts": np.sort(start[found]), "holding_days": spec.holding_days,
            }
        self._pair: dict[tuple[tuple, tuple], np.ndarray] = {}

    def n_targets(self, key: tuple) -> int:
        return len(self._sets[key]["ids"])

    def mask(self, a: tuple, b: tuple) -> np.ndarray:
        """Per target of a: whether it clashes with b's targets."""
        if (a, b) not in self._pair:
            sa, sb = self._sets[a], self._sets[b]
            has = sa["start"] >= 0
            starts = sb["sorted_starts"]
            # b's window [s, s + h_b - 1] (cut at the ticker's end) shares a day with a's
            # [start, end] if s lies in [start - h_b + 1, end] on the same ticker.
            lo = np.maximum(sa["start"][has] - sb["holding_days"] + 1, sa["ticker_start"][has])
            hit = np.zeros(len(sa["ids"]), dtype=bool)
            hit[has] = (np.searchsorted(starts, sa["end"][has], side="right")
                        - np.searchsorted(starts, lo, side="left")) > 0
            hit |= np.isin(sa["ids"], sb["ids"])
            self._pair[(a, b)] = hit
        return self._pair[(a, b)]

    def dropped(self, keys: list[tuple]) -> list[int]:
        """Per position in keys: targets dropped for clashing with another position."""
        out = []
        for i, a in enumerate(keys):
            hit = np.zeros(self.n_targets(a), dtype=bool)
            for j, b in enumerate(keys):
                if i != j:
                    hit |= self.mask(a, b)
            out.append(int(hit.sum()))
        return out


# --- Assignment -----------------------------------------------------------------

_COVERAGE_ATTEMPTS = 20
_ZERO_COST = (0, 0, 0, 0)


class _Packer:
    """Specs per planted universe under the rules in the module docstring."""

    def __init__(self, specs, index: ClashIndex, n_universes, per_universe, min_events,
                 max_dropped_share, seed):
        self.specs, self.index = specs, index
        self.n, self.per_universe, self.min_events = n_universes, per_universe, min_events
        self.max_dropped_share = max_dropped_share
        self.key_of = {s.spec_id: k for s, k in zip(specs, index.keys, strict=True)}
        self.rng = np.random.default_rng([seed, _PLAN_STREAM])
        self.rank = self._ranks()
        self.n_feature_types = len({s.feature_type for s in specs})
        # Specs that cannot share a universe even as a pair.
        self.conflicts = {
            a.spec_id: {b.spec_id for b in specs
                        if b.spec_id != a.spec_id and self.cost([a, b])[0]}
            for a in specs
        }

    def _ranks(self) -> dict[str, int]:
        order = self.rng.permutation(len(self.specs))
        return {s.spec_id: int(r) for s, r in zip(self.specs, order, strict=True)}

    def cost(self, members) -> tuple[int, int, int, int]:
        """(specs over a clash limit, repeated strengths, missing feature types,
        clashing targets). Compared lexicographically."""
        if not members:
            return _ZERO_COST
        keys = [self.key_of[m.spec_id] for m in members]
        dropped = self.index.dropped(keys)
        short = sum(self._over_limit(self.index.n_targets(k), d)
                    for k, d in zip(keys, dropped, strict=True))
        repeats = len(members) - len({m.strength for m in members})
        missing = min(len(members), self.n_feature_types) - len({m.feature_type
                                                                 for m in members})
        return short, repeats, missing, sum(dropped)

    def _over_limit(self, n_targets: int, n_dropped: int) -> bool:
        return (n_targets - n_dropped < self.min_events
                or n_dropped > self.max_dropped_share * n_targets)

    def _added(self, spec, members) -> tuple[int, ...]:
        before, after = self.cost(members), self.cost(members + [spec])
        return tuple(b - a for a, b in zip(before, after, strict=True))

    def fits(self, spec, members) -> bool:
        ids = {m.spec_id for m in members}
        return (len(members) < self.per_universe and spec.spec_id not in ids
                and not self.conflicts[spec.spec_id] & ids
                and self.cost(members + [spec])[0] == 0)

    def _place(self, spec, universes) -> bool:
        """Put spec in the fitting universe that is emptiest, then cheapest to add to."""
        options = [u for u in range(self.n) if self.fits(spec, universes[u])]
        if not options:
            return False
        best = min(options, key=lambda u: (len(universes[u]),
                                           *self._added(spec, universes[u]), u))
        universes[best].append(spec)
        return True

    def cover(self) -> list[list[PlantSpec]] | None:
        """One appearance of every spec, most conflicts first; None if it gets stuck.

        Ties in conflicts go by the seeded rank, then by a fresh seeded order on each
        later attempt.
        """
        for attempt in range(_COVERAGE_ATTEMPTS):
            ties = self.rank if attempt == 0 else self._ranks()
            order = sorted(self.specs,
                           key=lambda s: (-len(self.conflicts[s.spec_id]), ties[s.spec_id]))
            universes: list[list[PlantSpec]] = [[] for _ in range(self.n)]
            if all(self._place(s, universes) for s in order):
                return universes
        return None

    def fill(self, universes) -> None:
        """Fill free slots, each time with a fitting spec of the fewest appearances.

        A spec only gets another appearance while it has no more than the fewest any spec
        has, so appearances never differ by more than one.
        """
        counts = dict.fromkeys(self.rank, 0)
        for members in universes:
            for m in members:
                counts[m.spec_id] += 1
        while True:
            fewest = min(counts.values())
            options = [(u, s) for u in range(self.n) for s in self.specs
                       if counts[s.spec_id] == fewest and self.fits(s, universes[u])]
            if not options:
                return
            u, s = min(options, key=lambda o: (counts[o[1].spec_id],
                                               *self._added(o[1], universes[o[0]]),
                                               len(universes[o[0]]), self.rank[o[1].spec_id],
                                               o[0]))
            universes[u].append(s)
            counts[s.spec_id] += 1

    def improve(self, universes) -> None:
        """Swap specs between universes while a swap lowers the summed cost.

        A swap keeps every spec's appearances and never puts a spec twice in one
        universe. Passes run in a fixed order, so the result is deterministic.
        """
        costs = [self.cost(m) for m in universes]
        improved = True
        while improved:
            improved = False
            for u in range(self.n):
                for v in range(u + 1, self.n):
                    for i in range(len(universes[u])):
                        for j in range(len(universes[v])):
                            a, b = universes[u][i], universes[v][j]
                            if (b.spec_id in {m.spec_id for m in universes[u]}
                                    or a.spec_id in {m.spec_id for m in universes[v]}):
                                continue
                            new_u = universes[u][:i] + [b] + universes[u][i + 1:]
                            new_v = universes[v][:j] + [a] + universes[v][j + 1:]
                            cu, cv = self.cost(new_u), self.cost(new_v)
                            old = [x + y for x, y in zip(costs[u], costs[v], strict=True)]
                            if [x + y for x, y in zip(cu, cv, strict=True)] < old:
                                universes[u], universes[v] = new_u, new_v
                                costs[u], costs[v] = cu, cv
                                improved = True


def assign_specs(
    specs: list[PlantSpec],
    index: ClashIndex,
    n_universes: int,
    per_universe: int,
    min_events: int,
    seed: int,
    max_dropped_share: float = 1.0,
) -> list[list[PlantSpec]]:
    """Specs per planted universe, in planting order (see the module docstring)."""
    if n_universes == 0:
        return []
    if not specs:
        raise ValueError("no kept specs to plant")
    slots = n_universes * per_universe
    if len(specs) > slots:
        raise ValueError(
            f"{len(specs)} kept specs do not fit in {n_universes} planted universes x "
            f"{per_universe} specs = {slots} slots; raise n_planted to at least "
            f"{-(-len(specs) // per_universe)} or specs_per_universe"
        )
    packer = _Packer(specs, index, n_universes, per_universe, min_events, max_dropped_share,
                     seed)
    universes = packer.cover()
    if universes is None:
        n_conflicted = sum(bool(c) for c in packer.conflicts.values())
        raise ValueError(
            f"could not place every kept spec in {n_universes} planted universes with at "
            f"least plant_min_events={min_events} targets and at most "
            f"max_dropped_share={max_dropped_share:g} of them dropped per spec; raise "
            f"n_planted ({n_conflicted} of {len(specs)} specs cannot share a universe with "
            f"at least one other spec)"
        )
    packer.fill(universes)
    packer.improve(universes)
    if any(packer.cost(m)[0] for m in universes):  # improve() only lowers the cost
        raise AssertionError("a planted universe has a spec over the clash limits")
    return [sorted(m, key=lambda s: packer.rank[s.spec_id]) for m in universes]


def _universe_ids(experiment_id: str, n: int, seed: int) -> list[str]:
    """n opaque ids in a seeded random order."""
    order = np.random.default_rng([seed, _PLAN_STREAM, 1]).permutation(n)
    return [f"{experiment_id}-u{k:02d}" for k in order]


def plan_experiment(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    cfg: ExperimentConfig,
    benchmarks: pd.DataFrame | None = None,
) -> ExperimentPlan:
    """The plan for cfg on this data. benchmarks only enter the fingerprint."""
    cfg = cfg.resolved()
    if benchmarks is None:
        benchmarks = pd.DataFrame(columns=["date", "ticker", "ret"])
    if cfg.grid is not None:
        kept = grid_specs(cfg.grid, events, cfg.plant_min_events, cfg.holding_days,
                          cfg.base_seed)
        dropped = []
    else:
        kept, dropped = default_spec_grid(events, cfg.plant_min_events, cfg.holding_days,
                                          cfg.base_seed)
    if cfg.n_planted:
        index = ClashIndex(prices, events, kept)
        assigned = assign_specs(kept, index, cfg.n_planted, cfg.specs_per_universe,
                                cfg.plant_min_events, cfg.base_seed, cfg.max_dropped_share)
    else:
        assigned = []

    n_total = cfg.n_planted + cfg.n_placebo + int(cfg.include_real)
    ids = iter(_universe_ids(cfg.experiment_id, n_total, cfg.base_seed))
    universes, clashes = [], {}
    for i, specs in enumerate(assigned):
        seed = cfg.base_seed + i
        entry = UniverseEntry(
            universe_id=next(ids), universe_type="planted", seed=seed,
            holding_days=cfg.holding_days, min_events=cfg.plant_min_events,
            specs=[s.model_copy(update={"seed": seed}) for s in specs],
        )
        universes.append(entry)
        clashes[entry.universe_id] = sum(index.dropped([_target_key(s) for s in specs]))
    for i in range(cfg.n_placebo):
        universes.append(UniverseEntry(
            universe_id=next(ids), universe_type="placebo",
            seed=cfg.base_seed + PLACEBO_SEED_OFFSET + i, holding_days=cfg.holding_days,
            min_gap_days=cfg.min_gap_days,
        ))
    if cfg.include_real:
        universes.append(UniverseEntry(universe_id=next(ids), universe_type="real",
                                       holding_days=cfg.holding_days))

    if cfg.dev_guard is not None:  # a hard check on every plan, not just in tests
        seeds = {cfg.base_seed} | {u.seed for u in universes if u.seed is not None}
        check_dev_overlap(seeds, kept, cfg.dev_guard, events)

    appearances = {s.spec_id: 0 for s in kept}
    for specs in assigned:
        for s in specs:
            appearances[s.spec_id] += 1
    return ExperimentPlan(
        experiment_id=cfg.experiment_id, base_seed=cfg.base_seed, config=cfg,
        fingerprint=data_fingerprint(prices, benchmarks, events), kept_specs=kept,
        dropped_specs=dropped, appearances=appearances, planned_clashes=clashes,
        universes=universes,
    )


def plan_sha256(plan: ExperimentPlan) -> str:
    """SHA-256 of the canonical plan JSON (sorted keys, no whitespace)."""
    text = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_plan_sha256(cfg: ExperimentConfig) -> str | None:
    """The hash recorded in cfg.plan_sha256_file, or None if there is none yet."""
    if cfg.plan_sha256_file is None:
        return None
    path = REPO_ROOT / cfg.plan_sha256_file
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").split()[0]


def save_plan(plan: ExperimentPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=1), encoding="utf-8")


def load_plan(path: Path) -> ExperimentPlan:
    return ExperimentPlan.model_validate_json(path.read_text(encoding="utf-8"))
